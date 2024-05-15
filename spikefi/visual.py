from math import sqrt, prod
import matplotlib
import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np

from spikefi.core import CampaignData
import spikefi.fault as ff
from spikefi.utils.io import make_fig_filepath


CMAP = 'jet'


def _data_mapping(cmpn_data: CampaignData, layer: str = None, fault_model: ff.FaultModel = None) -> dict[tuple[str, ff.FaultModel], list[int]]:
    data_map: dict[tuple[str, ff.FaultModel], list[int]] = {}   # { (layer, fault model), [round index] }

    for lay, r_idxs in cmpn_data.rgroups.items():
        for r in r_idxs:
            round = cmpn_data.rounds[r]
            if len(round) > 1 or (layer and lay != layer):
                continue
            key = next(iter(round.keys()))

            if fault_model and fault_model not in key:
                continue

            data_map.setdefault(key, [])
            data_map[key].append(r)
    return data_map


def _shape_square(N: int) -> tuple[int, int]:
    x = int(sqrt(N))
    while N % x != 0:
        x -= 1

    return (x, int(N / x))


# TODO visual.bar
def bar() -> None:
    pass


# TODO visual.bar_comparative
def bar_comparative() -> None:
    pass


def colormap(format: str = 'svg') -> None:
    fig = plt.figure()
    fig.set_size_inches(fig.get_figwidth(), 1)

    cax = fig.add_axes([.05, .55, .9, .25])
    norm = matplotlib.colors.Normalize(0, 100)

    cbar = matplotlib.colorbar.Colorbar(cax, cmap=CMAP, norm=norm, orientation='horizontal', ticks=range(0, 101, 10))
    cbar.set_ticks(range(0, 100), minor=True)
    plt.xlabel("Classification Accuracy (%)")

    plot_path = make_fig_filepath(filename="colormap." + format.removeprefix('.'))
    plt.savefig(plot_path, transparent=True)


def heat(cmpn_data: CampaignData, layer: str = None, fault_model: ff.FaultModel = None,
         preserve_dim: bool = False, max_size: int = 512, title_suffix: str = None, format: str = 'svg') -> None:
    heat_max = max_size**2
    data_map = _data_mapping(cmpn_data, layer, fault_model)
    for (lay, fm), r_idxs in data_map.items():
        N = len(r_idxs)
        if N > heat_max:
            print("Cannot plot heat map for the following layer - fault model pair:")
            print((lay, fm))
            print(f"Reason: too many faults (>{heat_max}).")
            continue

        is_syn = fm.is_synaptic()
        shape = cmpn_data.layers_info.shapes_syn[lay] if is_syn else cmpn_data.layers_info.shapes_neu[lay]

        if N != prod(shape):
            plot_shape = (1, N)
        else:
            if is_syn:
                if shape[2] == 1 and shape[3] == 1:
                    plot_shape = (shape[0], shape[1])
                else:
                    plot_shape = (shape[0] * shape[1], shape[2] * shape[3])
            else:
                plot_shape = (shape[1] * shape[2], shape[0])

        if not preserve_dim or plot_shape[0] > sqrt(heat_max) or plot_shape[1] > sqrt(heat_max):
            plot_shape = _shape_square(N)

        perf = np.zeros(N)
        for i, r in enumerate(r_idxs):
            test_stats = cmpn_data.performance[r].testing
            perf[i] = test_stats.maxAccuracy

        fig = plt.figure(str((lay, fm)))

        hx = int(plot_shape[0] / 100.) + 1
        wx = int(plot_shape[1] / 100.) + 1
        fig.set_size_inches(wx * fig.get_figwidth(), hx * fig.get_figheight())

        pos = plt.imshow(perf.reshape(*plot_shape), cmap=CMAP,
                         origin='lower', vmin=0., vmax=1., interpolation='none',
                         extent=[0, plot_shape[1], 0, plot_shape[0]])

        pos.axes.set_xticks([1] + np.arange(10, plot_shape[1] + 1, 10).tolist())
        pos.axes.set_xticklabels([1] + np.arange(10, plot_shape[1] + 1, 10).tolist())
        pos.axes.set_xticks(np.arange(1, plot_shape[1]), minor=True)

        pos.axes.set_yticks([1] + np.arange(10, plot_shape[0] + 1, 10).tolist())
        pos.axes.set_yticklabels([1] + np.arange(10, plot_shape[0] + 1, 10).tolist())
        pos.axes.set_yticks(np.arange(1, plot_shape[0]), minor=True)

        pos.axes.tick_params(axis='both', which='both', length=0)
        pos.axes.grid(which='both', linestyle='-')

        if title_suffix is None and len(data_map) > 1:
            title_suffix = f"_{lay}_{fm.get_name()}{int(fm.args[0])}"
        elif title_suffix:
            title_suffix = "_" + title_suffix
        plot_path = make_fig_filepath(filename=f"{cmpn_data.name}_heat{title_suffix or ''}.{format.removeprefix('.')}")

        plt.savefig(plot_path, bbox_inches='tight', transparent=False)
