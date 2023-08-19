from .utils.quantization import Quantizer
from typing import List, Dict, Tuple, Callable
from torch import Tensor
import slayerSNN as snn

# TODO: Allow for selection of the fault's effectiveness in Time ?
# e.g., for either a neuron, or a synapse fault:
# set faulty output only for t_a:t_b in the tensor's 5th dim,
# while keeping the golden one for the rest of the output's timesteps


class FaultSite:
    def __init__(self, layer_name: str = None, dim0: int = None, chw: Tuple[int] = None) -> None:
        self.layer = layer_name
        self.dim0 = dim0
        self.set_chw(chw)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FaultSite):
            return False

        return self.layer == other.layer and self.dim0 == other.dim0 and self.get_chw() == other.get_chw()

    def set_chw(self, chw: Tuple[int]) -> None:
        if chw:
            assert len(chw) == 3

        self.channel, self.height, self.width = [chw[i] for i in range(3)] if chw else 3*[None]

    def get_chw(self) -> Tuple[int]:
        return (self.channel, self.height, self.width)

    def is_random(self) -> bool:
        return not self.layer

    def unroll(self) -> Tuple:
        return (self.dim0, self.channel, self.height, self.width, slice(None))

    @staticmethod
    def pos2str(x: int) -> str:
        return '*' if x is None else ':' if x == slice(None) else str(x)

    def key(self) -> Tuple:
        dim0_key = -1 if type(self.dim0) is slice else self.dim0
        return (self.layer, dim0_key, self.get_chw())

    def __repr__(self) -> str:
        return f"Fault Site: layer {self.layer or '*'} " + \
            f"at ({', '.join(list(map(FaultSite.pos2str, self.unroll()[0:4])))})"


class FaultModel:
    targets = {'output': 'z', 'weight': 'w', 'param': 'p'}

    def __init__(self, target: str, method: Callable[..., float], *args) -> None:
        assert target in FaultModel.targets.values(), "Invalid fault model target"

        self.target = target
        self.method = method
        self.args = args

        self.original = None
        self.perturbed = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FaultModel):
            return False

        return self.target == other.target and self.method == other.method and self.args == other.args

    def get_name(self) -> str:
        name = self.__class__.__name__
        return 'Custom' if name == FaultModel.__name__ else name

    def __repr__(self) -> str:
        s = f"Fault Model: '{self.get_name()}'\n"
        s += f"  - Target: {self.target}\n"
        s += f"  - Method: {self.method.__name__}\n"
        s += "  - Arguments:"
        for i, arg in enumerate(self.args):
            s += f"\n    ~ Arg {i+1}: {arg}"
        return s

    def perturb(self, original: float) -> float:
        self.original = original
        self.perturbed = self.method(original, *self.args)

        return self.perturbed

    def restore(self) -> float:
        tore = self.original
        self.original = None
        self.perturbed = None

        return tore

    def is_synaptic(self) -> bool:
        return self.target == 'w'

    def is_neuronal(self) -> bool:
        return not self.is_synaptic()

    def is_parametric(self) -> bool:
        return self.target == 'p'

    @staticmethod
    def set_value(_, value: float) -> float:
        return value

    @staticmethod
    def add_value(original: Tensor, value: float) -> Tensor:
        return original + value

    @staticmethod
    def mul_value(original: Tensor, value: float) -> Tensor:
        return original * value

    @staticmethod
    def qua_value(original: Tensor, qua: Quantizer) -> Tensor:
        return qua.dequantize(qua.quantize(original))

    @staticmethod
    def bfl_value(original: Tensor, qua: Quantizer, bit: int) -> Tensor:
        assert bit >= 0 and bit <= qua.precision

        return qua.dequantize(qua.quantize(original) ^ 2 ** bit)


class ParametricFaultModel(FaultModel):
    def __init__(self, param_name: str, param_method: Callable[..., float], *param_args) -> None:
        super().__init__('p', FaultModel.set_value)

        self.param_name = param_name
        self.param_method = param_method
        self.param_args = param_args

        self.param_original = None
        self.param_perturbed = None

        self.flayer = None

    def perturb_param(self, param_original: float) -> float:
        self.param_original = param_original
        self.param_perturbed = self.param_method(param_original, *self.param_args)

        return self.param_perturbed


# Neuron fault models
class DeadNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__('z', FaultModel.set_value, 0.)


class SaturatedNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__('z', FaultModel.set_value, 1.)


class ParametricNeuron(ParametricFaultModel):
    def __init__(self, param_name: str, percentage: float) -> None:
        super().__init__(param_name, FaultModel.mul_value, percentage)


# Synapse fault models
class DeadSynapse(FaultModel):
    def __init__(self) -> None:
        super().__init__('w', FaultModel.set_value, 0.)


class SaturatedSynapse(FaultModel):
    def __init__(self, satu: float) -> None:
        super().__init__('w', FaultModel.set_value, satu)


class BitflippedSynapse(FaultModel):
    def __init__(self, precision: int, wmin: float, wmax: float, bit: int) -> None:
        super().__init__('w', FaultModel.bfl_value, Quantizer(precision, wmin, wmax), bit)


class Fault:
    def __init__(self, model: FaultModel, sites: List[FaultSite] = [], occurrences: int = None) -> None:
        self.model = model
        self.sites = sites + [FaultSite() for _ in range(max(len(sites), occurrences or 1) - len(sites))]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fault):
            return False

        return self.model == other.model and \
            sorted(self.sites, key=FaultSite.key) == sorted(other.sites, key=FaultSite.key)

    def is_multiple(self) -> bool:
        return len(self) > 1

    def sort(self, reverse: bool = False) -> None:
        self.sites.sort(key=FaultSite.key, reverse=reverse)

    @staticmethod
    def sorted(fault: 'Fault', reverse: bool = False) -> 'Fault':
        return Fault(fault.model, sorted(fault.sites, key=FaultSite.key, reverse=reverse))

    def __len__(self) -> int:
        return len(self.sites)

    def __repr__(self) -> str:
        return f"Fault: '{self.model.get_name()}' @ {len(self)} site{'s' if self.is_multiple() else ''}"


class FaultRound:
    def __init__(self, lay_names: List[str]) -> None:
        self.round = {name: [] for name in lay_names}
        self.stats = snn.utils.stats()  # TODO: Replace with custom stats object ?
        self.keys = lay_names
        self.map_faults()

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

                self.fault_map[lay_name]['z'] |= f.model.is_neuronal()
                self.fault_map[lay_name]['p'] |= f.model.is_parametric()
                self.fault_map[lay_name]['w'] |= f.model.is_synaptic()

                try:
                    lay_fault = next(lay_fault for lay_fault in lay_faults if lay_fault.model == f.model)
                    lay_fault.sites.extend(lay_sites)
                except StopIteration:
                    lay_faults.append(Fault(f.model, lay_sites))

        for lay_name, lay_faults in self.round.items():
            for lay_fault in lay_faults:
                lay_fault.sort()

    def get_faults(self, layer_name: str = None, targets: str = None) -> List[Fault]:
        faults = self.round[layer_name] if layer_name else [f for lay_faults in self.round.values() for f in lay_faults]

        if targets:
            return [f for f in faults if f.model.target in targets]

        return faults

    def get_synapse_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, 'w')

    def get_neuron_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, 'zp')

    def get_parametric_faults(self, layer_name: str = None) -> List[Fault]:
        return self.get_faults(layer_name, 'p')

    def has_faults(self, layer_name: str = None, targets: str = None) -> bool:
        lay_names = [layer_name] if layer_name else self.keys
        if not targets:
            targets = 'zpw'

        return any(self.fault_map[lay_name][t] for lay_name in lay_names for t in targets)

    def has_synapse_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, 'w')

    def has_neuron_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, 'zp')

    def has_parametric_faults(self, layer_name: str = None) -> bool:
        return self.has_faults(layer_name, 'p')

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

        self.map_faults()

    def map_faults(self) -> Dict[str, Dict[str, bool]]:
        self.fault_map = {name: {t: False for t in FaultModel.targets.values()} for name in self.keys}

        for lay_faults in self.round.values():
            for f in lay_faults:
                for s in f.sites:
                    self.fault_map[s.layer]['z'] |= f.model.is_neuronal()
                    self.fault_map[s.layer]['p'] |= f.model.is_parametric()
                    self.fault_map[s.layer]['w'] |= f.model.is_synaptic()

        return self.fault_map

    def __contains__(self, fault: Fault) -> bool:
        return any(fault in inner_faults for inner_faults in self.round.values())

    def __repr__(self) -> str:
        return str(self.round)
