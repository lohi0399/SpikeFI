from typing import Tuple

import torch
from torch import Tensor


def q2i_dtype(qdtype: torch.dtype) -> torch.dtype:
    if qdtype is torch.quint8:
        idtype = torch.uint8
    elif qdtype is torch.qint8:
        idtype = torch.int8
    elif qdtype is torch.qint32:
        idtype = torch.int32
    else:
        raise AssertionError("The desired data type of returned tensor has to be \
                             one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32")

    return idtype


def quant_args_from_range(xmin: float | Tensor, xmax: float | Tensor,
                          dtype: torch.dtype) -> Tuple[Tensor, Tensor, torch.dtype]:
    if not torch.is_tensor(xmin):
        xmin = torch.tensor(xmin)
    if not torch.is_tensor(xmax):
        xmax = torch.tensor(xmax)
    xmin = xmin.float()
    xmax = xmax.float()

    assert xmin.size() == xmax.size()

    dt_info = torch.iinfo(q2i_dtype(dtype))
    qmin = dt_info.min
    qmax = dt_info.max

    scale = ((xmax - xmin) / (qmax - qmin))
    zero_point = torch.clip(qmin - xmin / scale, qmin, qmax).int()

    return scale, zero_point, dtype
