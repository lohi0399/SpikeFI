from copy import deepcopy
from enum import auto, Flag
from functools import reduce
from operator import or_
from typing import Callable, Iterable, List, Tuple

import torch
from torch import Tensor

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

from .utils.quantization import q2i_dtype, quant_args_from_range

# TODO: Allow for selection of the fault's effectiveness in Time ? (= transient faults?)
# e.g., for either a neuron, or a synapse fault:
# set faulty output only for t_a:t_b in the tensor's 5th dim,
# while keeping the golden one for the rest of the output's timesteps


class FaultSite:
    def __init__(self, layer_name: str = None, dim0: int | slice = None, chw: Tuple[int, int, int] = None) -> None:
        self.layer = layer_name
        self.dim0 = dim0
        self.set_chw(chw)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FaultSite):
            return False

        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"Fault Site: layer {self.layer or '*'} \
                 at ({', '.join(list(map(FaultSite.pos2str, self.unroll()[0:4])))})"

    def _key(self) -> Tuple:
        dim0_key = (self.dim0.start, self.dim0.stop, self.dim0.step) if isinstance(self.dim0, slice) else self.dim0
        return self.layer, dim0_key, self.get_chw()

    def get_chw(self) -> Tuple[int, int, int]:
        return self.channel, self.height, self.width

    def is_defined(self) -> bool:
        return not self.is_random() and self.dim0 is not None and all(dim is not None for dim in self.get_chw())

    def is_random(self) -> bool:
        return not self.layer

    def set_chw(self, chw: Tuple[int, int, int]) -> None:
        if chw:
            assert len(chw) == 3

        self.channel, self.height, self.width = [chw[i] for i in range(3)] if chw else 3 * [None]

    def unroll(self) -> Tuple[int | slice, int, int, int, slice]:
        # dim0 is integer for synaptic faults and either integer or slice for neuron faults
        return self.dim0, self.channel, self.height, self.width, slice(None)

    @staticmethod
    def pos2str(x: int) -> str:
        return '*' if x is None else ':' if x == slice(None) else str(x)


class FaultTarget(Flag):
    OUTPUT = Z = auto()
    WEIGHT = W = auto()
    PARAMETER = P = auto()

    @classmethod
    def all(cls):
        return reduce(or_, cls)


class FaultModel:
    def __init__(self, target: FaultTarget, method: Callable[..., float | Tensor], *args) -> None:
        assert len(target) == 1
        self.target = target
        self.method = method
        self.args = args

        self.original = {}
        self.perturbed = {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FaultModel):
            return False

        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        s = f"Fault Model: '{self.get_name()}'\n\
              - Target: {self.target}\n\
              - Method: {self.method.__name__}\n\
              - Arguments:"
        for i, arg in enumerate(self.args):
            s += f"\n    ~ Arg {i+1}: {arg}"

        return s

    def _key(self) -> Tuple:
        return self.target, self.method, self.args

    def get_name(self) -> str:
        name = self.__class__.__name__
        return 'Custom' if name == FaultModel.__name__ else name

    def is_neuronal(self) -> bool:
        return self.target in FaultTarget.PARAMETER | FaultTarget.OUTPUT

    def is_parametric(self) -> bool:
        return self.target is FaultTarget.PARAMETER

    def is_synaptic(self) -> bool:
        return self.target is FaultTarget.WEIGHT

    def is_perturbed(self, site: FaultSite) -> bool:
        return not site and site in self.original and site in self.perturbed

    # Omitting the site means no restoration will be needed
    def perturb(self, original: float | Tensor, site: FaultSite = None) -> float | Tensor:
        perturbed = self.method(original, *self.args)

        if site:
            self.original[site] = original
            self.perturbed[site] = perturbed

        return perturbed

    def restore(self, site: FaultSite) -> float | Tensor:
        if not self.is_perturbed(site):
            return None

        self.perturbed.pop(site)
        return self.original.pop(site)

    @staticmethod
    def set_value(_, value: float | Tensor) -> float | Tensor:
        return value

    @staticmethod
    def add_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
        return original + value

    @staticmethod
    def mul_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
        return original * value

    @staticmethod
    def qua_value(original: float | Tensor,
                  scale: float | Tensor, zero_point: int | Tensor, dtype: torch.dtype) -> Tensor:
        return torch.dequantize(torch.quantize_per_tensor(original, scale, zero_point, dtype))

    @staticmethod
    def bfl_value(original: float | Tensor, bit: int,
                  scale: float | Tensor, zero_point: int | Tensor, dtype: torch.dtype) -> Tensor:
        idt_info = torch.iinfo(q2i_dtype(dtype))
        assert bit >= 0 and bit < idt_info.bits

        q = torch.quantize_per_tensor(original, scale, zero_point, dtype)
        return torch.dequantize(q ^ 2 ** bit)


class ParametricFaultModel(FaultModel):
    def __init__(self, param_name: str, param_method: Callable[..., float | Tensor], *param_args) -> None:
        super().__init__(FaultTarget.PARAMETER, FaultModel.set_value)

        self.param_name = param_name
        self.param_method = param_method
        self.param_args = param_args

        self.param_original = None
        self.param_perturbed = None
        self.flayer = None

    def __repr__(self) -> str:
        s = super().__repr__() + f"\n\
              - Parameter Name: '{self.param_name}'\n\
              - Parameter Method: {self.param_method}\n\
              - Parameter Arguments:"
        for i, arg in enumerate(self.param_args):
            s += f"\n    ~ Arg {i+1}: {arg}"

        return s

    def _key(self) -> Tuple:
        return super()._key() + (self.param_name, self.param_method, self.param_args)

    def is_param_perturbed(self) -> bool:
        return self.flayer is not None

    # Parametric faults have only one site, so original and perturbed variables are scalars
    def param_perturb(self, slayer: spikeLayer) -> None:
        self.flayer = deepcopy(slayer)

        self.param_original = self.flayer.neuron[self.param_name]
        self.param_perturbed = self.param_method(self.param_original, *self.param_args)
        self.flayer.neuron[self.param_name] = self.param_perturbed

    def param_restore(self) -> None:
        tore = self.param_original

        self.param_original = None
        self.param_perturbed = None
        self.flayer = None

        return tore


# Neuron fault models
class DeadNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.OUTPUT, FaultModel.set_value, 0.)


class SaturatedNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.OUTPUT, FaultModel.set_value, 1.)


class ParametricNeuron(ParametricFaultModel):
    def __init__(self, param_name: str, percentage: float) -> None:
        super().__init__(param_name, FaultModel.mul_value, percentage)


# Synapse fault models
class DeadSynapse(FaultModel):
    def __init__(self) -> None:
        super().__init__(FaultTarget.WEIGHT, FaultModel.set_value, 0.)


class SaturatedSynapse(FaultModel):
    def __init__(self, satu: float) -> None:
        super().__init__(FaultTarget.WEIGHT, FaultModel.set_value, satu)


class BitflippedSynapse(FaultModel):
    def __init__(self, bit: int, wmin: float, wmax: float, quant_dtype: torch.dtype) -> None:
        super().__init__(FaultTarget.WEIGHT, FaultModel.bfl_value, bit, *quant_args_from_range(wmin, wmax, quant_dtype))


class Fault:
    def __init__(self, model: FaultModel, sites: FaultSite | Iterable[FaultSite] = [], random_sites_num: int = 0) -> None:
        self.model = model

        if not sites and not random_sites_num:
            self.sites_defined = set()
            self.sites_pending = [FaultSite()]
            return

        sites_ = [] if not sites else [sites] if isinstance(sites, FaultSite) else sites
        self.sites_defined = {s for s in sites_ if s.is_defined()}
        self.sites_pending = [s for s in sites_ if not s.is_defined()]
        if random_sites_num:
            self.sites_pending.extend([FaultSite() for _ in range(random_sites_num)])

    def __add__(self, other: 'Fault') -> FaultSite():
        assert self.model == other.model

        all_sites_defined = set(self.sites_defined)
        all_sites_defined.update(other.sites_defined)
        all_sites = list(all_sites_defined) + self.sites_pending + other.sites_pending

        return Fault(deepcopy(self.model), all_sites)

    def __bool__(self) -> bool:
        return self is not None and bool(self.sites_defined)

    def __contains__(self, site: FaultSite) -> bool:
        return site in self.sites_defined

    def __eq__(self, other: object) -> bool:
        # and collections.Counter(self.sites_pending) == collections.Counter(other.sites_pending)
        return isinstance(other, Fault) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __len__(self) -> int:
        return len(self.sites_defined)

    def __iter__(self):
        return iter(self.sites_defined)

    def __next__(self):
        return next(self.sites_defined)

    def __repr__(self) -> str:
        return f"Fault: '{self.model.get_name()}' @ {len(self)} site{'s' if self.is_multiple() else ''}"

    def _key(self) -> Tuple:
        return self.model, frozenset(self.sites_defined)

    def add_site(self, site: FaultSite) -> None:
        if site is None:
            return

        if site.is_defined():
            self.sites_defined.add(site)
        else:
            self.sites_pending.append(site)

    def get_sites(self) -> List[FaultSite]:
        return list(self.sites_defined) + self.sites_pending

    def is_complete(self) -> bool:
        return not self.sites_pending

    def is_multiple(self) -> bool:
        return len(self) > 1

    def update_sites(self, sites: Iterable[FaultSite]) -> None:
        if not sites:
            return

        self.sites_defined.update({s for s in sites if s.is_defined()})
        self.sites_pending.extend([s for s in sites if not s.is_defined()])

    def refresh(self, discard_duplicates: bool = False) -> None:
        to_remove = []
        for s in self.sites_pending:
            if not s.is_defined():
                continue

            if self.sites_defined.isdisjoint({s}) or discard_duplicates:
                to_remove.append(s)
            self.sites_defined.add(s)

        for s in to_remove:
            self.sites_pending.remove(s)


class FaultRound:
    def __init__(self) -> None:
        self.stats = snn.utils.stats()
        self.round = {}
        self.keys = set()
        self.fault_map = {}

    def __bool__(self) -> bool:
        return bool(self.round)

    def __contains__(self, fault: Fault) -> bool:
        for fset in self.round.values():
            if fault in fset:
                return True

        return False

    def __len__(self) -> int:
        return len(self.round)

    # TODO: Verify iterable of FaultRound
    def __iter__(self):
        self._faults = self.round.values()
        return iter(self._faults)

    def __next__(self):
        return next(self._faults)

    def __repr__(self) -> str:
        s = 'Fault Round: {'
        for lay_name, fault_set in self.round.items():
            s += f"\n  @ {lay_name}: {fault_set}"

        return s + '\n}'

    def _create_fault_map(self) -> None:
        self.keys = set(self.round.keys())
        self.fault_map = {name: {t: False for t in FaultTarget} for name in self.keys}

        for lay_fset in self.round.values():
            for f in lay_fset:
                for s in f:
                    self.fault_map[s.layer][FaultTarget.Z] |= f.model.is_neuronal()
                    self.fault_map[s.layer][FaultTarget.P] |= f.model.is_parametric()
                    self.fault_map[s.layer][FaultTarget.W] |= f.model.is_synaptic()

    def get_faults(self, layer_name: str = None, targets: FaultTarget = FaultTarget(0)) -> List[Fault]:
        if layer_name and layer_name not in self.keys:
            return []

        faults = self.round[layer_name] if layer_name else [f for lay_fset in self.round.values() for f in lay_fset]

        if targets:
            return [f for f in faults if f.model.target in targets]

        return faults

    def get_neuron_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, FaultTarget.OUTPUT | FaultTarget.PARAMETER)

    def get_parametric_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, FaultTarget.PARAMETER)

    def get_synapse_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, FaultTarget.WEIGHT)

    def has_faults(self, layer_name: str = None, targets: FaultTarget = FaultTarget.all()) -> bool:
        if layer_name and layer_name not in self.keys:
            return False

        lay_names = [layer_name] if layer_name else self.keys

        return any(self.fault_map[lay_name][t] for lay_name in lay_names for t in targets)

    def has_neuron_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, FaultTarget.OUTPUT | FaultTarget.PARAMETER)

    def has_parametric_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, FaultTarget.PARAMETER)

    def has_synapse_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, FaultTarget.WEIGHT)

    def insert(self, faults: Fault | Iterable[Fault]) -> None:
        faults_ = [] if not faults else [faults] if isinstance(faults, Fault) else faults

        for f in faults_:
            # TODO: Check here if FaultModel is already in the round
            # If not, create an empty Fault with this FaultModel
            # and use it directly in the sites loop (without try-catch)
            for s in f:
                if s.layer not in self.round:
                    self.round[s.layer] = set()

                # Make sure that each parametric fault has a single site,
                # since fault model depends on the neuron's output
                if f.model.is_parametric():
                    f.model.args = tuple()
                    self.round[s.layer].add(Fault(deepcopy(f.model), s))
                    continue

                try:
                    lay_fault = next(lay_fault for lay_fault in self.round[s.layer] if lay_fault.model == f.model)
                    lay_fault.add_site(s)
                except StopIteration:
                    self.round[s.layer].add(Fault(f.model, s))

        self._create_fault_map()

    def remove(self, faults: Fault | Iterable[Fault]) -> None:
        faults_ = [] if not faults else [faults] if isinstance(faults, Fault) else faults

        for f in faults_:
            if not f:
                continue
            for lay_name, lay_fset in self.round.items():
                sites_to_remove = [s for s in f if s.layer == lay_name]
                if not sites_to_remove:
                    continue

                try:
                    lay_fault = next(lay_fault for lay_fault in lay_fset if lay_fault.model == f.model)
                    for s in sites_to_remove:
                        if s in lay_fault:
                            lay_fault.sites_defined.discard(s)

                    # FIXME: Fault is no longer indexed by set because it and its hash are changed
                    if not lay_fault:
                        lay_fset.discard(lay_fault)
                except StopIteration:
                    continue

        self._create_fault_map()
