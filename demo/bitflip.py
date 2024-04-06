import os
import torch

import spikefi as sfi
import demo as cs

# LSB: bit 0
# MSB: bit B
B = 8
qdtype = torch.quint8
layer_name = 'SF2'

fnetname = cs.get_fnetname(trial='4')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

cmpn = sfi.Campaign(net, cs.shape_in, net.slayer,
                    name=fnetname.removesuffix('.pt') + f'_synapse_bitflip_{layer_name}')

layer = getattr(net, layer_name)
wmin = layer.weight.min().item()
wmax = layer.weight.max().item()

fmodels = []
for b in range(B):
    fmodels.append(sfi.fm.BitflippedSynapse(b, wmin, wmax, qdtype))

cmpn.inject_complete(fmodels, [layer_name], fault_sampling_k=sfi.visual.MAX_SYN_NUM)

print(cmpn)
cmpn.run(cs.test_loader)
print(f"{cmpn.duration : .2f} secs")

cmpn.save()

for b in range(B):
    sfi.visual.plot_synaptic(cmpn.export(),
                             fault_model=fmodels[b],
                             plot_name=cmpn.name.replace('bitflip', f'bitflip{b}'))
