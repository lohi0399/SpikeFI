import os
import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs

layer_name = 'SF2'

fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

for o in range(5):
    cmpn = sfi.Campaign(net, cs.shape_in, net.slayer,
                        name=fnetname.removesuffix('.pt') + f"_neuron_dead_{layer_name or 'ALL'}_O{o}")

    cmpn.inject_complete([sfi.fm.DeadNeuron()], [layer_name] if layer_name else [])
    print(cmpn.name)

    cmpn.run(cs.test_loader, error=snn.loss(cs.net_params).to(cmpn.device),
             opt=sfi.CampaignOptimization(o))
    print(f"{cmpn.duration : .2f} secs")

    cmpn.save()
