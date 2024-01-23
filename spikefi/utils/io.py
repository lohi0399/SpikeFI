import os


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


def rename_if_multiple(filename: str, parentdir: str) -> str:
    split_fname = os.path.splitext(filename)[0]
    trial = len([f for f in os.listdir(parentdir) if split_fname[0] in f and f.endswith(split_fname[1])])

    return split_fname[0] + (str(trial) if trial else '') + split_fname[1]
