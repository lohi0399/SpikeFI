from collections.abc import Callable, Iterable
from copy import copy, deepcopy
from enum import auto, Flag
from functools import reduce
from operator import or_

import torch
from torch import Tensor

import slayerSNN as snn
from slayerSNN.slayer import spikeLayer

from .utils.quantization import q2i_dtype, quant_args_from_range


class FaultSite:
    def __init__(self, layer_name: str = None, position: tuple[int | slice, int, int, int] = (None,) * 4) -> None:
        # pos0 is integer for synaptic faults and either integer or slice for neuron faults
        self.layer = layer_name

        if position is not None:
            assert len(position) == 4, 'Position tuple must have a length of 4'
        self.position = position or (None,) * 4

    def __bool__(self) -> bool:
        return self is not None and self.is_defined()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FaultSite) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "Fault Site @ layer '" + (self.layer or '*') + "' " \
            + "(" + ", ".join(list(map(FaultSite.pos2str, self.position))) + ")"

    def _key(self) -> tuple:
        pos0 = self.position[0]
        pos0_key = (pos0.start, pos0.stop, pos0.step) if isinstance(pos0, slice) else pos0

        return self.layer, pos0_key, self.position[1:]

    def is_defined(self) -> bool:
        return bool(self.layer) and all(pos is not None for pos in self.position)

    def unroll(self) -> tuple[int | slice, int, int, int, slice]:
        return self.position + (slice(None),)

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
        assert len(target) == 1, 'Fault Model must have exactly 1 Fault Target'
        self.target = target
        self.method = method
        self.args = args

        self.original = {}
        self.perturbed = {}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FaultModel) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"Fault Model: '{self.get_name()}'\n" \
            + f"  - Target: {self.target}\n" \
            + f"  - Method: {self.method.__name__}\n" \
            + f"  - Arguments: {self.args}"

    def __str__(self) -> str:
        return f"Fault Model '{self.get_name()}': {self.target}, {self.method.__name__}"

    def _key(self) -> tuple:
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
        return site is not None and site in self.original and site in self.perturbed

    # Omitting the site means no restoration will be needed
    def perturb(self, original: float | Tensor, site: FaultSite, *new_args) -> float | Tensor:
        args_ = new_args or self.args
        perturbed = self.method(original, *args_)

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
        assert bit >= 0 and bit < idt_info.bits, 'Invalid bit position to flip'

        q = torch.quantize_per_tensor(original, scale, zero_point, dtype)
        return torch.dequantize(q ^ 2 ** bit)


class ParametricFaultModel(FaultModel):
    def __init__(self, param_name: str, param_method: Callable[..., float | Tensor], *param_args) -> None:
        super().__init__(FaultTarget.PARAMETER, FaultModel.set_value, dict())

        self.param_name = param_name
        self.param_method = param_method
        self.param_args = param_args

        self.param_original = None
        self.param_perturbed = None
        self.flayer = None

    def __repr__(self) -> str:
        return super().__repr__() + "\n" \
            + f"  - Parameter Name: '{self.param_name}'\n" \
            + f"  - Parameter Method: {self.param_method.__name__}\n" \
            + f"  - Parameter Arguments: {self.param_args}"

    def __str__(self) -> str:
        return super().__str__() + f" | Parametric: '{self.param_name}', {self.param_method.__name__}"

    def _key(self) -> tuple:
        return self.target, self.method, self.param_name, self.param_method, self.param_args

    def is_param_perturbed(self) -> bool:
        return self.flayer is not None

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

    def perturb(self, original: float | Tensor, site: FaultSite) -> float | Tensor:
        return super().perturb(original, site, self.args[0][site])


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
    def __init__(self, model: FaultModel, sites: FaultSite | Iterable[FaultSite] = None, random_sites_num: int = 0) -> None:
        self.model = model
        self.sites = set()
        self.sites_pending = []

        if sites is None and not random_sites_num:
            return

        if isinstance(sites, Iterable):
            self.update_sites(sites)
        else:
            self.add_site(sites)

        if random_sites_num:
            self.sites_pending.extend([FaultSite() for _ in range(random_sites_num)])

    def __add__(self, other: 'Fault') -> FaultSite():
        assert self.model == other.model, 'Only two Faults with the same Fault Model can be added'
        return Fault(deepcopy(self.model), self.get_sites(include_pending=True) + other.get_sites(include_pending=True))

    def __bool__(self) -> bool:
        return self is not None and bool(self.sites)

    def __contains__(self, site: FaultSite) -> bool:
        return site in self.sites

    def __eq__(self, other: object) -> bool:
        # Equality is checked so as the Faults were injected "as is",
        # therefore it does not take into account the pending Fault Sites,
        # as they are considered random sites and are not yet defined (finalized)
        return isinstance(other, Fault) and self.sites == other.sites

    def __len__(self) -> int:
        return len(self.sites)

    def __repr__(self) -> str:
        s = f"Fault '{self.model.get_name()}' @ {self.sites or '0 sites'}"
        if self.sites_pending:
            s += f" (+{len(self.sites_pending)} pending)"

        return s

    def __str__(self) -> str:
        s = f"Fault '{self.model.get_name()}' @ "
        if len(self) == 1:
            f = next(iter(self.sites))
            s += str(f).split('@ ')[-1]
        else:
            s += f"{len(self)} sites"

        if self.sites_pending:
            s += f" (+{len(self.sites_pending)} pending)"

        return s

    def add_site(self, site: FaultSite) -> None:
        if site is None:
            return

        if site.is_defined():
            self.sites.add(site)
        else:
            self.sites_pending.append(site)

    def breakdown(self) -> list['Fault']:
        separated_faults = []
        for s in self.get_sites(include_pending=True):
            separated_faults.append(Fault(deepcopy(self.model), s))

        return separated_faults

    def get_sites(self, include_pending: bool = False) -> list[FaultSite]:
        return list(self.sites) + (self.sites_pending if include_pending else [])

    def is_complete(self) -> bool:
        return not self.sites_pending

    def is_multiple(self) -> bool:
        return len(self) > 1

    def refresh(self, discard_duplicates: bool = False) -> None:
        newly_defined = []
        for s in self.sites_pending:
            if not s.is_defined():
                continue

            if self.sites.isdisjoint({s}) or discard_duplicates:
                newly_defined.append(s)
            self.sites.add(s)

        for s in newly_defined:
            self.sites_pending.remove(s)

    def update_sites(self, sites: Iterable[FaultSite]) -> None:
        if sites is None:
            return
        if not isinstance(sites, Iterable):
            raise TypeError(f"'{type(sites).__name__}' object is not iterable")

        for s in sites:
            self.add_site(s)


class FaultRound(dict):
    def __init__(self, *args, **kwargs) -> None:
        self.stats = snn.utils.stats()
        if 'faults' in kwargs or (args and isinstance(args[0], Iterable) and all(isinstance(el, Fault) for el in args[0])):
            super().__init__()
            self.insert_many(kwargs.get('faults', args[0]))
        else:
            super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return self._info(verbose=True)

    def __str__(self) -> str:
        return self._info(verbose=False)

    def _info(self, verbose: bool) -> str:
        sfunc = repr if verbose else str
        s = 'Fault Round: {'
        for key, fault in self.items():
            s += f"\n  '{key[0]}': {sfunc(fault)}"

        return s + '\n}'

    def extract(self, fault: Fault) -> None:
        if not fault:
            return

        for s in copy(fault.sites):
            key = (s.layer, fault.model)
            f = self.get(key)
            if f is not None:
                f.sites.discard(s)
                if not f:
                    del self[key]

    def extract_many(self, faults: Iterable[Fault]) -> None:
        if faults is None:
            return
        if not isinstance(faults, Iterable):
            raise TypeError(f"'{type(faults).__name__}' object is not iterable")

        for f in faults:
            self.extract(f)

    def insert(self, fault: Fault) -> None:
        if not fault:
            return

        for s in fault.sites:
            key = (s.layer, fault.model)
            self.setdefault(key, Fault(deepcopy(fault.model)))
            self[key].add_site(s)

    def insert_many(self, faults: Iterable[Fault]) -> None:
        if faults is None:
            return
        if not isinstance(faults, Iterable):
            raise TypeError(f"'{type(faults).__name__}' object is not iterable")

        for f in faults:
            self.insert(f)

    def search(self, layer_name: str = None, target: FaultTarget = None) -> list[Fault]:
        lay_keys = [k for k in self.keys() if k[0] == layer_name] if layer_name else self.keys()
        if target and target != FaultTarget.all():
            lay_keys = [k for k in lay_keys if k[1].target in target]

        return [self[k] for k in lay_keys]

    def search_neuronal(self, layer_name: str = None) -> list[Fault]:
        return self.search(layer_name, FaultTarget.OUTPUT | FaultTarget.PARAMETER)

    def search_parametric(self, layer_name: str = None) -> list[Fault]:
        return self.search(layer_name, FaultTarget.PARAMETER)

    def search_synaptic(self, layer_name: str = None) -> list[Fault]:
        return self.search(layer_name, FaultTarget.WEIGHT)
