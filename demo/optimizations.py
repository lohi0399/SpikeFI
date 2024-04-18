from copy import copy
import os
import torch
from torch.utils.data import DataLoader

import slayerSNN as snn

import spikefi as sfi
import demo as cs

fm_type = sfi.fm.DeadNeuron
L = ['']
B = [1]
K = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 500]  # 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 500
O = [0, 1]  # noqa E741

fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for lay_name in L:
    rounds = []  # Accumulated fault injection rounds
    rghist = {}  # Round groups histogram - frequencies of faults per layer
    for k in sorted(K):
        for bs in B:
            test_loader = DataLoader(dataset=cs.test_set, batch_size=bs, shuffle=cs.shuffle)
            for o in O:
                cmpn_name = fnetname.removesuffix('.pt') + f"_neuron_dead_{lay_name or 'ALL'}_bs{bs}_k{k}_O{o}"
                cmpn = sfi.Campaign(net, cs.shape_in, net.slayer, name=cmpn_name)
                cmpn.rounds = rounds

                # Accumulation of faults if k changes for the same layer
                # Actual number of (new) faults to be injected
                k_actual = k - len(cmpn.rounds)
                if k_actual > 0:
                    shapes = cmpn.layers_info.shapes_syn if fm_type().is_synaptic() else cmpn.layers_info.shapes_neu

                    if lay_name:
                        # Try to inject k faults
                        cmpn.inject_complete(fm_type(), [lay_name], fault_sampling_k=k_actual)

                        # Fault hyper-sampling
                        while k - len(cmpn.rounds) > 0:
                            cmpn.then_inject([sfi.ff.Fault(fm_type(), sfi.ff.FaultSite(lay_name))])
                    else:
                        # Equally distribute faults across layers
                        k_lay = int(k_actual / len(cmpn.layers_info.get_injectables()))
                        for lay in cmpn.layers_info.get_injectables():
                            n_lay = len(cmpn.inject_complete(fm_type(), [lay], fault_sampling_k=k_lay))
                            rghist.setdefault(lay, 0)
                            rghist[lay] += n_lay

                        # Inject remaining faults
                        while k - len(cmpn.rounds) > 0:
                            min_lay = min(rghist, key=rghist.get)
                            cmpn.then_inject([sfi.ff.Fault(fm_type(), sfi.ff.FaultSite(min_lay))])
                            rghist[min_lay] += 1

                print(cmpn.name)
                cmpn.run(test_loader, error=snn.loss(cs.net_params).to(cmpn.device), opt=sfi.CampaignOptimization(o))
                print(f"{cmpn.duration : .2f} secs")

                rounds = copy(cmpn.rounds)

                cmpn.save()
