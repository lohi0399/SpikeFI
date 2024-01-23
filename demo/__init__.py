__all__ = ["plot", "train",
           "Dataset", "Network"]

import os

from torch.utils.data import DataLoader

import slayerSNN as snn

# Configuration parameters (modify depending on application)
CASE_STUDY = 'nmnist-deep'
DO_ENABLED = False
EPOCHS_NUM = 100
OUT_DIR = 'out/net'

_error = ValueError(f"Case study '{CASE_STUDY}' not added.")

# To work on a new case study:
#   - Create a new import case

# Case study imports
if 'nmnist' in CASE_STUDY:
    fyamlname = 'nmnist'
    from demo.nets.nmnist import NMNISTDataset as Dataset

    if CASE_STUDY == 'nmnist-deep':
        from demo.nets.nmnist import NMNISTNetwork as Network
    elif CASE_STUDY == 'nmnist-lenet':
        from demo.nets.nmnist import LeNetNetwork as Network
    else:
        raise _error
else:
    raise _error

# No changes needed after this line
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)

base_fname = f"{CASE_STUDY}{'-do' if DO_ENABLED else ''}"
trial = str(len([f for f in os.listdir(OUT_DIR) if base_fname in f and f.endswith('.pt')]) or '')

fnetname = f"{base_fname}_net{trial}.pt"
fstaname = f"{base_fname}_stats{trial}.pkl"
ffigname = f"{base_fname}_train{trial}.svg"

net_params = snn.params(f'demo/config/{fyamlname}.yaml')

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
