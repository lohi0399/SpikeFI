import torch
from torch import Tensor

import spikefi.fault as ff
import spikefi.utils.quantization as qua


# Fault Model Functions
def set_value(_, value: float | Tensor) -> float | Tensor:
    return value


def add_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
    return original + value


def mul_value(original: float | Tensor, value: float | Tensor) -> float | Tensor:
    return original * value


def qua_value(original: float | Tensor, scale: float | Tensor, zero_point: int | Tensor, dtype: torch.dtype) -> Tensor:
    return torch.dequantize(torch.quantize_per_tensor(original, scale, zero_point, dtype))


def bfl_value(original: float | Tensor, bit: int, scale: float | Tensor, zero_point: int | Tensor, dtype: torch.dtype) -> Tensor:
    idt_info = torch.iinfo(qua.q2i_dtype(dtype))
    assert bit >= 0 and bit < idt_info.bits, 'Invalid bit position to flip'

    q = torch.quantize_per_tensor(original, scale, zero_point, dtype)
    return torch.dequantize(q ^ 2 ** bit)


# Neuron fault models
class DeadNeuron(ff.FaultModel):
    def __init__(self) -> None:
        super().__init__(ff.FaultTarget.OUTPUT, set_value, 0.)


class SaturatedNeuron(ff.FaultModel):
    def __init__(self) -> None:
        super().__init__(ff.FaultTarget.OUTPUT, set_value, 1.)


class ParametricNeuron(ff.ParametricFaultModel):
    def __init__(self, param_name: str, percentage: float) -> None:
        super().__init__(param_name, mul_value, percentage)


# Synapse fault models
class DeadSynapse(ff.FaultModel):
    def __init__(self) -> None:
        super().__init__(ff.FaultTarget.WEIGHT, set_value, 0.)


class SaturatedSynapse(ff.FaultModel):
    def __init__(self, satu: float) -> None:
        super().__init__(ff.FaultTarget.WEIGHT, set_value, satu)


class BitflippedSynapse(ff.FaultModel):
    def __init__(self, bit: int, wmin: float, wmax: float, quant_dtype: torch.dtype) -> None:
        super().__init__(ff.FaultTarget.WEIGHT, bfl_value, bit, *qua.quant_args_from_range(wmin, wmax, quant_dtype))
