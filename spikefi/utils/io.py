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


def calculate_trial(filename: str, parentdir: str) -> int:
    fname, extension = os.path.splitext(filename)
    fnames = [f.removesuffix(extension) for f in os.listdir(parentdir)
              if fname in f and f.endswith(extension)]

    if not fnames:
        return 0

    trial_matches = [re.search(r' \(\d+\)$', f) for f in fnames]

    return max([int(m.group().strip(' ()')) if m else 0 for m in trial_matches]) + 1


def rename_if_multiple(filename: str, parentdir: str) -> str:
    t = calculate_trial(filename, parentdir)
    if t == 0:
        return filename

    fname, extension = os.path.splitext(filename)

    return fname + f" ({t})" + extension
