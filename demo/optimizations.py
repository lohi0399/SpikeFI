from math import prod
import os
import torch
from torch.utils.data import DataLoader

import slayerSNN as snn

import spikefi as sfi
import demo as cs

layer_name = ''
fm_type = sfi.fm.DeadNeuron
B = [1]
K = [0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 500, 1000]
O = [0, 1]  # noqa E741

fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for bs in B:
    test_loader = DataLoader(dataset=cs.test_set, batch_size=bs, shuffle=cs.shuffle, num_workers=cs.test_loader.num_workers)
    for k in K:
        for o in O:
            cmpn_name = fnetname.removesuffix('.pt') + f"_neuron_dead_{layer_name or 'ALL'}_bs{bs}_k{k}_O{o}"
            cmpn = sfi.Campaign(net, cs.shape_in, net.slayer, name=cmpn_name)

            cmpn.inject_complete(fm_type(), [layer_name] if layer_name else [], fault_sampling_k=k)

            # Fault Super Sampling
            shapes = cmpn.layers_info.shapes_syn if fm_type().is_synaptic() else cmpn.layers_info.shapes_neu
            if layer_name:
                N = prod(shapes[layer_name])
            else:
                N = sum([prod(shape) if cmpn.layers_info.is_injectable(lay) else 0 for lay, shape in shapes.items()])

            for i in range(int(k / N)):
                for j in range((k % N) or 1):
                    cmpn.then_inject([sfi.ff.Fault(fm_type, sfi.ff.FaultSite(layer_name))])

            print(cmpn.name)
            cmpn.run(test_loader, error=snn.loss(cs.net_params).to(cmpn.device), opt=sfi.CampaignOptimization(o))
            print(f"{cmpn.duration : .2f} secs")

            cmpn.save()
