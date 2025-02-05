import os
import torch

from spikefi.models import DeadNeuron, ParametricNeuron, SaturatedSynapse, BitflippedSynapse
from spikefi.fault import FaultSite, Fault
from spikefi.core import Campaign
from spikefi import visual

import demo as cs
from demo import shape_in, test_loader

# Initialization
fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

cmpn = Campaign(net, shape_in, net.slayer)

fx = Fault(DeadNeuron(), FaultSite('SF2'))
fy = Fault(SaturatedSynapse(10), FaultSite('SF1'))
fz = Fault(ParametricNeuron('theta', 0.5))

cmpn.inject([fx])
cmpn.then_inject([fy, fz])

# Run
cmpn.run(test_loader)

for perf in cmpn.performance:
    print(perf.testing.maxAccuracy * 100.0)

# Save & Visualize
cmpn.save()
visual.bar(cmpn.export())

# Reset and run new campaign layer-wise
cmpn.eject()

layer = getattr(net, 'SF2')
wmin = layer.weight.min().item()
wmax = layer.weight.max().item()

cmpn.inject_complete(BitflippedSynapse(7, wmin, wmax, torch.uint8), ['SF2'])

cmpn.run(test_loader)

visual.heat([cmpn.export()])
