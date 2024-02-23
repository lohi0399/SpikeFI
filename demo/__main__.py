import os
import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs


fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, fnetname))
net.eval()

cmpn = sfi.Campaign(net, (2, 34, 34), net.slayer, name=fnetname.removesuffix('.pt') + '_neuron_dead_SF1_O4')

s1 = sfi.ff.FaultSite(layer_name='SF1')
s2 = sfi.ff.FaultSite(layer_name='SF2')
s3 = sfi.ff.FaultSite('SC3', (slice(None), 2, 2, 0))
f1 = sfi.ff.Fault(sfi.fm.DeadNeuron(), s1)
f2 = sfi.ff.Fault(sfi.fm.SaturatedNeuron(), s2)
f3 = sfi.ff.Fault(sfi.fm.DeadNeuron(), s3)
f4 = sfi.ff.Fault(sfi.fm.SaturatedSynapse(2.), random_sites_num=1)
f5 = sfi.ff.Fault(sfi.fm.DeadNeuron(), random_sites_num=2)
f6 = sfi.ff.Fault(sfi.fm.ParametricNeuron('theta', 0.1), [sfi.ff.FaultSite('SC1'), sfi.ff.FaultSite('SC1')])
f7 = sfi.ff.Fault(sfi.fm.ParametricNeuron('theta', 0.1), sfi.ff.FaultSite('SC2'))
f8 = sfi.ff.Fault(sfi.fm.SaturatedNeuron(), [sfi.ff.FaultSite('SC3')])

cmpn.inject([f1, f2, f3])
cmpn.inject([f1])
cmpn.then_inject([f6, f7])
cmpn.then_inject([f8])

cmpn.eject()

cmpn.inject_complete([sfi.fm.DeadNeuron()], ['SF1'])

print(cmpn)
cmpn.run(cs.test_loader, error=snn.loss(cs.net_params).to(cmpn.device), opt=sfi.CampaignOptimization.O4)

cmpn.save()

print(f"{cmpn.duration : .2f} secs")
