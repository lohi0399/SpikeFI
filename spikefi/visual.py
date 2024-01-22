from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os

from spikefi.core import CampaignData
import spikefi.fault as ff
from spikefi.utils.io import make_filepath


MAX_SYN_NUM = 250000
PLOT_FORMAT = '.png'


# TODO: Implement
def plot_neuronal(cmpn_data: CampaignData, fault_model: ff.FaultModel,
                  accuracy: bool = True, plot_name: str = None) -> None:
    pass


def plot_synaptic(cmpn_data: CampaignData, fault_model: ff.FaultModel,
                  accuracy: bool = True, plot_name: str = None) -> None:
    if not fault_model.is_synaptic():
        print("Use only with synaptic fault models.")
        return

    os.makedirs(os.path.join(os.getcwd(), 'fig'), exist_ok=True)

    for layer, r_idxs in cmpn_data.rgroups.items():
        syn_shape = cmpn_data.layers_info.shapes_syn[layer]
        syn_num = np.prod(syn_shape)
        if syn_num > MAX_SYN_NUM:
            print(f"Cannot plot synaptic faults for layer {layer}: too many synapses (>{MAX_SYN_NUM}).")
            continue

        if syn_shape[2] == 1 and syn_shape[3] == 1:
            plot_shape = (syn_shape[0], syn_shape[1])
        else:
            plot_shape = (syn_shape[0] * syn_shape[1], syn_shape[2] * syn_shape[3])

        if plot_shape[0] > 500 or plot_shape[1] > 500:
            x = int(sqrt(syn_num))
            while syn_num % x != 0:
                x -= 1

            plot_shape = (x, int(syn_num / x))

        assert syn_num == len(r_idxs), 'Insufficient number of rounds. Cannot plot due to missing data.'

        perf = np.zeros(syn_shape)
        for r in r_idxs:
            round = cmpn_data.rounds[r]
            faults = [f for f in round.search(layer) if f.model == fault_model]
            for fault in faults:
                for site in fault.sites:
                    # TODO: Check difference between best and min/max stats
                    test_stats = cmpn_data.performance[r].testing
                    perf[site.position] = test_stats.maxAccuracy if accuracy else test_stats.minloss

        fig = plt.figure(layer)

        hx = int(plot_shape[0] / 100.) + 1
        wx = int(plot_shape[1] / 100.) + 1
        fig.set_size_inches(wx * fig.get_figwidth(), hx * fig.get_figheight())

        pos = plt.imshow(perf.reshape(*plot_shape), cmap='jet',
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

        if not plot_name:
            plot_name = f"synaptic_{layer}_{fault_model.get_name()}_{fault_model.args[0]}"

        plot_path = make_filepath(out_dir='fig', out_fname=plot_name + PLOT_FORMAT)

        plt.savefig(plot_path, bbox_inches='tight', transparent=False)

    return fig
