__all__ = ["neuromorphic", "nmnist",
           "CSDataset", "CSNetwork"]

# Configuration parameters (modify depending on application)
case_study = 'nmnist-lenet'
to_train = True
do_enable = False
epochs = 100
out_dir = 'out/net'

# To work on a new case study:
#   - Add its name in the list
#   - Make a new import case

assert case_study in ['nmnist-deep', 'nmnist-lenet', 'gesture'], \
    f'Case study {case_study} value is not valid. Please import it.'

# Case study imports
if 'nmnist' in case_study:
    fyamlname = 'nmnist'
    from example.nmnist import NMNISTDataset as CSDataset

    if case_study == 'nmnist-deep':
        from example.nmnist import NMNISTNetwork as CSNetwork
    elif case_study == 'nmnist-lenet':
        from example.nmnist import LeNetNetwork as CSNetwork
