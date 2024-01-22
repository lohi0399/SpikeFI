import os


OUT_DIR = 'out'
FIG_DIR = 'fig'


def make_filepath(filename: str, parentdir: str = '') -> str:
    return os.path.join(os.getcwd(), parentdir, filename)


def make_out_filepath(filename: str) -> str:
    return make_filepath(filename, OUT_DIR)


def make_fig_filepath(filename: str) -> str:
    return make_filepath(filename, FIG_DIR)
