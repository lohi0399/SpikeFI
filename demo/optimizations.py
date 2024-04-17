from copy import copy
import os
import torch
from torch.utils.data import DataLoader

import slayerSNN as snn

import spikefi as sfi
import demo as cs

fm_type = sfi.fm.DeadNeuron
L = ['SF2', 'SF1', 'SC3', 'SC2', 'SC1']
B = [1]
K = [30]  # 0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 500
O = [0, 1, 2, 3, 4]  # noqa E741

fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for lay_name in L:
    for k in K:
        rounds = []
        for bs in B:
            test_loader = DataLoader(dataset=cs.test_set, batch_size=bs, shuffle=cs.shuffle)
            for o in O:
                cmpn_name = fnetname.removesuffix('.pt') + f"_neuron_dead_{lay_name or 'ALL'}_bs{bs}_k{k}_O{o}"
                cmpn = sfi.Campaign(net, cs.shape_in, net.slayer, name=cmpn_name)

                if rounds:
                    cmpn.rounds = rounds
                else:
                    shapes = cmpn.layers_info.shapes_syn if fm_type().is_synaptic() else cmpn.layers_info.shapes_neu

                    if lay_name:
                        # Try to inject k faults
                        cmpn.inject_complete(fm_type(), [lay_name], fault_sampling_k=k)
                    else:
                        k_lay = int(k / len(cmpn.layers_info.get_injectables()))
                        # Equally distribute faults across layers
                        for ln in cmpn.layers_info.get_injectables():
                            cmpn.inject_complete(fm_type(), [ln], fault_sampling_k=k_lay)

                    # Fault hyper-sampling
                    for _ in range(k - len(cmpn.rounds)):
                        cmpn.then_inject([sfi.ff.Fault(fm_type(), sfi.ff.FaultSite(lay_name))])

                    rounds = copy(cmpn.rounds)

                print(cmpn.name)
                cmpn.run(test_loader, error=snn.loss(cs.net_params).to(cmpn.device), opt=sfi.CampaignOptimization(o))
                print(f"{cmpn.duration : .2f} secs")

                cmpn.save()
