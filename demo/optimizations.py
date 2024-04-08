from math import prod
import os
import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs

layer_name = 'SF2'
k = 100

fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for o in range(5):
    cmpn = sfi.Campaign(net, cs.shape_in, net.slayer,
                        name=fnetname.removesuffix('.pt') + f"_neuron_dead_{layer_name or 'ALL'}_O{o}_k{k}")

    cmpn.inject_complete([sfi.fm.DeadNeuron()], [layer_name] if layer_name else [],
                         fault_sampling_k=k)

    # Fault Super Sampling
    N = prod(cmpn.layers_info.shapes_neu[layer_name])
    for i in range(int(k / N)):
        for j in range((k % N) or 1):
            cmpn.then_inject([sfi.ff.Fault(sfi.fm.DeadNeuron(), sfi.ff.FaultSite(layer_name))])

    print(cmpn.name)

    cmpn.run(cs.test_loader, error=snn.loss(cs.net_params).to(cmpn.device),
             opt=sfi.CampaignOptimization(o))
    print(f"{cmpn.duration : .2f} secs")

    cmpn.save()
