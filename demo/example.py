import os
import torch

from spikefi.models import SaturatedNeuron, DeadNeuron, DeadSynapse, BitflippedSynapse
from spikefi.fault import FaultSite, Fault
from spikefi.core import Campaign
from spikefi import visual

import demo as cs
from demo import shape_in, test_loader


# Initialization
fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

# Fault Round 1
site1 = FaultSite(layer_name='SF2')
model1 = DeadNeuron()
fault1 = Fault(model1, site1)

# Fault Round 2
fault2 = Fault(DeadSynapse(), FaultSite(layer_name='SC1'))
fault3 = Fault(SaturatedNeuron(), random_sites_num=4)

# FI campaign initialization
cmpn = Campaign(net, shape_in, net.slayer)
cmpn.inject([fault1])
cmpn.then_inject([fault2, fault3])

# Run
cmpn.run(test_loader)
for perf in cmpn.performance:
    print(perf.testing.maxAccuracy)

# Reset and run new campaign layer-wise
cmpn.eject()
cmpn.inject_complete(BitflippedSynapse(), layer_names=['SF2'])
cmpn.run(test_loader)
visual.heat(cmpn.export())
