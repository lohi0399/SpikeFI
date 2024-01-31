import os
import re


OUT_DIR = 'out'
RES_DIR = 'out/res'
FIG_DIR = 'out/fig'


def make_filepath(filename: str, parentdir: str = '') -> str:
    return os.path.join(os.getcwd(), parentdir, filename)


def make_out_filepath(filename: str) -> str:
    return make_filepath(filename, OUT_DIR)


def make_fig_filepath(filename: str) -> str:
    return make_filepath(filename, FIG_DIR)


def make_res_filepath(filename: str) -> str:
    return make_filepath(filename, RES_DIR)


def calculate_trial(filename: str, parentdir: str) -> str:
    split_fname = os.path.splitext(filename)
    fnames = [f.removesuffix(split_fname[1]) for f in os.listdir(parentdir)
              if split_fname[0] in f and f.endswith(split_fname[1])]

    if not fnames:
        return ''

    trial_matches = [re.search(r'\d+$', f) for f in fnames]

    return str(max([int(m.group()) if m else 0 for m in trial_matches]) + 1)


def rename_if_multiple(filename: str, parentdir: str) -> str:
    split_fname = os.path.splitext(filename)

    return split_fname[0] + calculate_trial(filename, parentdir) + split_fname[1]
