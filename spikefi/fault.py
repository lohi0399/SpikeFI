from .utils.quantization import Quantizer
from typing import List, Tuple, Callable

# TODO: Allow for selection of the fault's effectiveness in Time ?
# e.g., for either a neuron, or a synapse fault:
# set faulty output only for t_a:t_b in the tensor's 5th dim,
# while keeping the golden one for the rest of the output's timesteps


class FaultSite:
    def __init__(self, layer: int = None, dim0: int = None, chw: Tuple[int] = None) -> None:
        self.layer = layer
        self.dim0 = dim0
        self.chw = chw
        self.channel, self.height, self.width = [chw[i] for i in range(3)] if chw else 3*[None]

    def __repr__(self) -> str:
        s = "Fault Site:\n"
        s += f"  - Layer #: {self.layer or '-'}\n"
        s += f"  - Position: ({self.dim0 or '-'}, {self.channel or '-'}, {self.height or '-'}, {self.width or '-'})"
        return s


class FaultModel:
    targets = {'input': 'a', 'output': 'z', 'weight': 'w', 'param': 'p'}

    def __init__(self, target: str, method: Callable[..., float], *args) -> None:
        assert target in FaultModel.targets.values(), "Invalid fault model target"

        self.target = target
        self.method = method
        self.args = args

        self.is_synaptic = target == 'w'

        self.original = None
        self.perturbed = None

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if name == FaultModel.__name__:
            name = "Custom"

        s = f"Fault Model: '{name}'\n"
        s += f"  - Target: {self.target}\n"
        s += f"  - Method: {self.method.__name__}\n"
        s += f"  - Arguments: {str(*self.args)}"
        return s

    def perturb(self, original: float) -> float:
        self.original = original
        self.perturbed = self.method(original, *self.args)

        return self.perturbed

    def is_synaptic(self) -> bool:
        return self.target == 'w'

    def is_neuronal(self) -> bool:
        return not self.is_synaptic()

    def is_parametric(self) -> bool:
        return self.target == 'p'

    @staticmethod
    def set_value(original: float, value: float) -> float:
        return value

    @staticmethod
    def add_value(original: float, value: float) -> float:
        return original + value

    @staticmethod
    def mul_value(original: float, value: float) -> float:
        return original * value

    @staticmethod
    def qua_value(original: float, qua: Quantizer) -> float:
        return qua.dequantize(qua.quantize(original))

    @staticmethod
    def bfl_value(original: float, qua: Quantizer, bit: int) -> float:
        assert bit >= 0 and bit <= qua.precision

        return qua.dequantize(qua.quantize(original) ^ 2 ** bit)


# Neuron fault models
class DeadNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__('z', FaultModel.set_value, 0.)


class SaturatedNeuron(FaultModel):
    def __init__(self) -> None:
        super().__init__('z', FaultModel.set_value, 1.)


class ParametricNeuron(FaultModel):
    def __init__(self, param: str, percentage: float) -> None:
        super().__init__('p', FaultModel.mul_value, percentage)
        self.param = param


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
        self.occurences = max(len(sites), occurrences if occurrences else 1)
        self.sites = sites + (self.occurences - len(sites)) * [FaultSite()]

    def is_multiple(self) -> bool:
        return self.occurences > 1

    def __repr__(self) -> str:
        s = f"+ {self.model}\n"
        if self.is_multiple():
            s += f"+ Multiple Fault\n+ Occurences: {self.occurences}"
        else:
            s += "Single Fault"

        return s
