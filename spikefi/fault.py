from enum import auto, Flag
from functools import reduce
from operator import or_
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import Tensor
import slayerSNN as snn

from .utils.quantization import q2i_dtype, quant_args_from_range

# TODO: Allow for selection of the fault's effectiveness in Time ?
# e.g., for either a neuron, or a synapse fault:
# set faulty output only for t_a:t_b in the tensor's 5th dim,
# while keeping the golden one for the rest of the output's timesteps


class FaultSite:
    def __init__(self, layer_name: str = None, dim0: int = None, chw: Tuple[int, int, int] = None) -> None:
        self.layer = layer_name
        self.dim0 = dim0
        self.set_chw(chw)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, FaultSite):
            return False

        return self.layer == __value.layer and self.dim0 == __value.dim0 and self.get_chw() == __value.get_chw()

    def __repr__(self) -> str:
        return f"Fault Site: layer {self.layer or '*'} " + \
            f"at ({', '.join(list(map(FaultSite.pos2str, self.unroll()[0:4])))})"

    def get_chw(self) -> Tuple[int, int, int]:
        return self.channel, self.height, self.width

    def is_random(self) -> bool:
        return not self.layer

    def key(self) -> Tuple[str, int, Tuple[int, int, int]]:
        dim0_key = -1 if type(self.dim0) is slice else self.dim0
        return self.layer, dim0_key, self.get_chw()

    def set_chw(self, chw: Tuple[int, int, int]) -> None:
        if chw:
            assert len(chw) == 3

        self.channel, self.height, self.width = [chw[i] for i in range(3)] if chw else 3*[None]

    def unroll(self) -> Tuple[Any, int, int, int, slice]:
        # dim0 is integer for synaptic faults and either integer or slice for neuron faults
        return self.dim0, self.channel, self.height, self.width, slice(None)

    @staticmethod
    def pos2str(x: int) -> str:
        return '*' if x is None else ':' if x == slice(None) else str(x)


class FaultTarget(Flag):
    OUTPUT = Z = auto()
    WEIGHT = W = auto()
    PARAM = P = auto()

    @classmethod
    def all(cls):
        return reduce(or_, cls)


class FaultModel:
    def __init__(self, target: FaultTarget, method: Callable[..., float | Tensor], *args) -> None:
        assert len(target) == 1
        self.target = target
        self.method = method
        self.args = args

        self.original = None
        self.perturbed = None

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, FaultModel):
            return False

        return self.target == __value.target and self.method == __value.method and self.args == __value.args

    def __repr__(self) -> str:
        s = f"Fault Model: '{self.get_name()}'\n"
        s += f"  - Target: {self.target}\n"
        s += f"  - Method: {self.method.__name__}\n"
        s += "  - Arguments:"
        for i, arg in enumerate(self.args):
            s += f"\n    ~ Arg {i+1}: {arg}"
        return s

    def get_name(self) -> str:
        name = self.__class__.__name__
        return 'Custom' if name == FaultModel.__name__ else name

    def is_neuronal(self) -> bool:
        return self.target in FaultTarget.PARAM | FaultTarget.OUTPUT

    def is_parametric(self) -> bool:
        return self.target is FaultTarget.PARAM

    def is_synaptic(self) -> bool:
        return self.target is FaultTarget.WEIGHT

    def perturb(self, original: float | Tensor) -> float | Tensor:
        self.original = original
        self.perturbed = self.method(original, *self.args)

        return self.perturbed

    def restore(self) -> float | Tensor:
        tore = self.original
        self.original = None
        self.perturbed = None

        return tore

    @staticmethod
    def _pre_method_assert(original: float | Tensor, value: float | Tensor) -> None:  # Not used
        numel1 = original.numel() if torch.is_tensor(original) else 1
        numel2 = value.numel() if torch.is_tensor(value) else 1

        assert numel1 == numel2

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
        super().__init__(FaultTarget.PARAM, FaultModel.set_value)

        self.param_name = param_name
        self.param_method = param_method
        self.param_args = param_args

        self.param_original = None
        self.param_perturbed = None

        self.flayer = None

    def perturb_param(self, param_original: float | Tensor) -> float | Tensor:
        self.param_original = param_original
        self.param_perturbed = self.param_method(param_original, *self.param_args)

        return self.param_perturbed

    def restore_param(self) -> float | Tensor:
        tore = self.param_original
        self.param_original = None
        self.param_perturbed = None

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
    def __init__(self, model: FaultModel, sites: List[FaultSite] = [], occurrences: int = None) -> None:
        self.model = model
        self.sites = sites + [FaultSite() for _ in range(max(len(sites), occurrences or 1) - len(sites))]

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Fault):
            return False

        return self.model == __value.model and \
            sorted(self.sites, key=FaultSite.key) == sorted(__value.sites, key=FaultSite.key)

    def __len__(self) -> int:
        return len(self.sites)

    def __repr__(self) -> str:
        return f"Fault: '{self.model.get_name()}' @ {len(self)} site{'s' if self.is_multiple() else ''}"

    def is_multiple(self) -> bool:
        return len(self) > 1

    def sort(self, reverse: bool = False) -> None:
        self.sites.sort(key=FaultSite.key, reverse=reverse)

    @staticmethod
    def sorted(fault: 'Fault', reverse: bool = False) -> 'Fault':
        return Fault(fault.model, sorted(fault.sites, key=FaultSite.key, reverse=reverse))


class FaultRound:
    def __init__(self, lay_names: List[str]) -> None:
        self.round = {name: [] for name in lay_names}
        self.stats = snn.utils.stats()  # TODO: Replace with custom stats object ?
        self.keys = lay_names
        self._create_fault_map()

    def __contains__(self, fault: Fault) -> bool:
        return any(fault in inner_faults for inner_faults in self.round.values())

    def __repr__(self) -> str:
        return str(self.round)

    def _create_fault_map(self) -> Dict[str, Dict[FaultTarget, bool]]:
        self.fault_map = {name: {t: False for t in FaultTarget} for name in self.keys}

        for lay_faults in self.round.values():
            for f in lay_faults:
                for s in f.sites:
                    self.fault_map[s.layer][FaultTarget.Z] |= f.model.is_neuronal()
                    self.fault_map[s.layer][FaultTarget.P] |= f.model.is_parametric()
                    self.fault_map[s.layer][FaultTarget.W] |= f.model.is_synaptic()

        return self.fault_map

    def get_faults(self, layer_name: str = None, targets: FaultTarget = FaultTarget(0)) -> List[Fault]:
        faults = self.round[layer_name] if layer_name else [f for lay_faults in self.round.values() for f in lay_faults]

        if targets:
            return [f for f in faults if f.model.target in targets]

        return faults

    def get_neuron_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, FaultTarget.OUTPUT | FaultTarget.PARAM)

    def get_parametric_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, FaultTarget.PARAM)

    def get_synapse_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, FaultTarget.WEIGHT)

    def has_faults(self, layer_name: str = None, targets: FaultTarget = FaultTarget.all()) -> bool:
        lay_names = [layer_name] if layer_name else self.keys

        return any(self.fault_map[lay_name][t] for lay_name in lay_names for t in targets)

    def has_neuron_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, FaultTarget.OUTPUT | FaultTarget.PARAM)

    def has_parametric_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, FaultTarget.PARAM)

    def has_synapse_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, FaultTarget.WEIGHT)

    def insert(self, faults: List[Fault]) -> None:
        for f in faults:
            if f.model.is_parametric():
                # Parametric faults have one site only, since fault model depends on the neuron's output
                for s in f.sites:
                    self.round[s.layer].append(Fault(f.model, [s]))
                continue

            for lay_name, lay_faults in self.round.items():
                lay_sites = [s for s in f.sites if s.layer == lay_name]
                if not lay_sites:
                    continue

                self.fault_map[lay_name][FaultTarget.Z] |= f.model.is_neuronal()
                self.fault_map[lay_name][FaultTarget.P] |= f.model.is_parametric()
                self.fault_map[lay_name][FaultTarget.W] |= f.model.is_synaptic()

                try:
                    lay_fault = next(lay_fault for lay_fault in lay_faults if lay_fault.model == f.model)
                    lay_fault.sites.extend(lay_sites)
                except StopIteration:
                    lay_faults.append(Fault(f.model, lay_sites))

        for _, lay_faults in self.round.items():
            for lay_fault in lay_faults:
                lay_fault.sort()

    def remove(self, faults: List[Fault]) -> None:
        for f in faults:
            for lay_name, lay_faults in self.round.items():
                lay_sites = [s for s in f.sites if s.layer == lay_name]
                if not lay_sites:
                    continue

                try:
                    lay_fault = next(lay_fault for lay_fault in lay_faults if lay_fault.model == f.model)
                    for lay_site in lay_sites:
                        if lay_site in lay_fault.sites:
                            lay_fault.sites.remove(lay_site)

                    if not lay_fault.sites:
                        lay_faults.remove(lay_fault)
                except StopIteration:
                    continue

        self._create_fault_map()
