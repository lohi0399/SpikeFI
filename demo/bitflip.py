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
                    name=fnetname.removesuffix('.pt') + f"_synapse_bitflip_{layer_name or 'ALL'}")

layer = getattr(net, layer_name)
wmin = layer.weight.min().item()
wmax = layer.weight.max().item()

for b in range(B):
    cmpn.inject_complete(sfi.fm.BitflippedSynapse(b, wmin, wmax, qdtype),
                         [layer_name] if layer_name else [], fault_sampling_k=250**2)

print(cmpn)
cmpn.run(cs.test_loader)
print(f"{cmpn.duration : .2f} secs")

cmpn.save()

sfi.visual.heat(cmpn.export())
