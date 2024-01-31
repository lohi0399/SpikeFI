__all__ = ["plot", "train",
           "Dataset", "Network"]

import os

from torch.utils.data import DataLoader

import slayerSNN as snn

import spikefi as sfi


# Configuration parameters (modify depending on application)
CASE_STUDY = 'nmnist-lenet'
DO_ENABLED = False
OUT_DIR = 'out/net'

_exception = ValueError(f"Case study '{CASE_STUDY}' not added.")

# To work on a new case study:
#   - Create a new import case

# Case study imports
if 'nmnist' in CASE_STUDY:
    fyamlname = 'nmnist.yaml'
    from demo.nets.nmnist import NMNISTDataset as Dataset

    if CASE_STUDY == 'nmnist-deep':
        from demo.nets.nmnist import NMNISTNetwork as Network
    elif CASE_STUDY == 'nmnist-lenet':
        from demo.nets.nmnist import LeNetNetwork as Network
    else:
        raise _exception
else:
    raise _exception

# No changes needed after this line
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)
base_fname = f"{CASE_STUDY}{'-do' if DO_ENABLED else ''}"

net_params = snn.params(f'demo/config/{fyamlname}')

train_set = Dataset(
    data_path=net_params['training']['path']['dir_train'],
    samples_file=net_params['training']['path']['list_train'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
train_loader = DataLoader(dataset=train_set, batch_size=12, shuffle=False, num_workers=4)

test_set = Dataset(
    data_path=net_params['training']['path']['dir_test'],
    samples_file=net_params['training']['path']['list_test'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
test_loader = DataLoader(dataset=test_set, batch_size=12, shuffle=False, num_workers=4)

trial_def = sfi.utils.io.calculate_trial(base_fname + '_.pt', OUT_DIR)


def get_fnetname(trial: str = None) -> str:
    return f"{base_fname}_net{trial or trial_def}.pt"


def get_fstaname(trial: str = None) -> str:
    return f"{base_fname}_stats{trial or trial_def}.pkl"


def get_ffigname(trial: str = None) -> str:
    return f"{base_fname}_train{trial or trial_def}.svg"
