from difflib import SequenceMatcher
from math import prod, sqrt
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from spikefi.core import CampaignData
import spikefi.fault as ff
from spikefi.utils.io import make_fig_filepath


CMAP = 'jet'


def _data_mapping(cmpns_data: list[CampaignData], layer: str = None,
                  fault_model: ff.FaultModel = None) -> dict[tuple[str, ff.FaultModel], dict[int, list[int]]]:
    data_map: dict[tuple[str, ff.FaultModel], dict[int, list[int]]] = {}   # { (layer, fault model): { campaign index, [round index] } }

    for cmpn_idx, cmpn_data in enumerate(cmpns_data):
        for lay, r_idxs in cmpn_data.rgroups.items():
            for r in r_idxs:
                round = cmpn_data.rounds[r]
                if len(round) > 1 or (layer and lay != layer):
                    continue
                key = next(iter(round.keys()))

                if fault_model and fault_model not in key:
                    continue

                data_map.setdefault(key, {})
                data_map[key].setdefault(cmpn_idx, [])
                data_map[key][cmpn_idx].append(r)

    return data_map


def _title_plot(cmpns_data: list[CampaignData], data_map: dict,
                model_friendly: str, plot_type: str, title_suffix: str, format: str) -> str:
    title_def = ""

    if len(data_map) == 1 and plot_type == "heat":
        fm = next(iter(data_map.keys()))[1]
        title_def = "_" + str(int(fm.args[0]))
    elif len(data_map) > 1:
        model = next(iter(data_map.keys()))[1]
        one_m = True
        for _, fm in data_map.keys():
            one_m &= fm == model

        if one_m:
            title_def = "" if model_friendly else f"_{model.get_name()}{int(fm.args[0])}"
        else:
            title_def = "_comparative"

    if model_friendly:
        model_friendly = "_" + model_friendly.strip('_')
    if title_suffix:
        title_suffix = "_" + title_suffix.strip('_')

    cmpn_name = cmpns_data[0].name
    for cmpn_data in cmpns_data:
        match = SequenceMatcher(None, cmpn_name, cmpn_data.name).find_longest_match()
        cmpn_name = cmpn_name[match.a:match.a + match.size]

    return f"{cmpn_name.strip('_')}{model_friendly or ''}{title_def}_{plot_type}{title_suffix or ''}.{format.strip('.')}"


def _shape_square(N: int) -> tuple[int, int]:
    x = int(sqrt(N))
    while N % x != 0:
        x -= 1

    return (x, int(N / x))


def bar(cmpns_data: list[CampaignData],
        model_friendly: str = None, title_suffix: str = None, format: str = 'svg') -> Figure:
    data_map = _data_mapping(cmpns_data)

    offset_mult: dict[str, int] = {}
    layers = sorted(list(set(key[0] for key in data_map.keys())))

    fig, ax = plt.subplots()
    colormap = plt.get_cmap(CMAP)
    width = 3 / len(data_map)
    space = width / 5

    for (lay, fm), cmpn_dict in data_map.items():
        offset_mult.setdefault(lay, 0)

        N = 0
        perf = []
        for cmpn_idx, r_idxs in cmpn_dict.items():
            N += len(r_idxs)
            for r in r_idxs:
                test_stats = cmpns_data[cmpn_idx].performance[r].testing
                perf.append(test_stats.maxAccuracy * 100.0)
        perf = np.array(perf)

        groups_freq = np.bincount(perf.round().astype(int), minlength=101)
        groups_cent = groups_freq / N * 100.0

        bottom = 0.0
        for i in range(101):
            offset = (width + space) * offset_mult[lay]
            b = ax.bar(layers.index(lay) + offset, groups_cent[i], width,
                       bottom=bottom, color=colormap(i/100.0))

            bottom += groups_cent[i]

        ax.bar_label(b, labels=[fm.get_name()[:4] + "."], rotation=90,
                     color='white', padding=-30)

        offset_mult[lay] += 1

    ax.set_ylabel('Faults (%)')
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel('Layers')
    ax.set_xticks([i + (width/2 + space/2) * (offset_mult[lay]-1) for i, lay in enumerate(layers)], layers)

    plot_path = make_fig_filepath(_title_plot(cmpns_data, data_map, model_friendly, "bar", title_suffix, format))
    plt.savefig(plot_path, bbox_inches='tight', transparent=False)

    return fig


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


def heat(cmpns_data: list[CampaignData], layer: str = None, fault_model: ff.FaultModel = None,
         preserve_dim: bool = False, max_area: int = 512**2, show_axes: bool = True,
         model_friendly: str = None, title_suffix: str = None, format: str = 'svg') -> list[Figure]:
    figs = []
    data_map = _data_mapping(cmpns_data, layer, fault_model)
    for (lay, fm), cmpn_dict in data_map.items():
        N = 0
        perf = []
        local_cmpns_data = []
        for cmpn_idx, r_idxs in cmpn_dict.items():
            N += len(r_idxs)
            local_cmpns_data.append(cmpns_data[cmpn_idx])

            for r in r_idxs:
                test_stats = cmpns_data[cmpn_idx].performance[r].testing
                perf.append(test_stats.maxAccuracy)
        perf = np.array(perf)

        if N > max_area:
            print("Cannot plot heat map for the following layer - fault model pair:")
            print((lay, fm))
            print(f"Reason: too many faults (>{max_area}).")
            continue

        is_syn = fm.is_synaptic()
        if is_syn:
            shape = local_cmpns_data[0].layers_info.shapes_syn[lay]
        else:
            shape = local_cmpns_data[0].layers_info.shapes_neu[lay]

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

        if not preserve_dim or plot_shape[0] > sqrt(max_area) or plot_shape[1] > sqrt(max_area):
            plot_shape = _shape_square(N)

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

        if not show_axes:
            pos.axes.set_xticklabels([])
            pos.axes.set_yticklabels([])

        plot_path = make_fig_filepath(_title_plot(local_cmpns_data, {(lay, fm): cmpn_dict}, model_friendly, "heat", title_suffix, format))
        plt.savefig(plot_path, bbox_inches='tight', transparent=False)

        figs.append(fig)

    return figs
