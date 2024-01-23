import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs

net: cs.Network = torch.load(f"{cs.OUT_DIR}/{cs.CASE_STUDY}{'-do' if cs.DO_ENABLED else ''}.pt")
net.eval()

cmpn = sfi.Campaign(net, (2, 34, 34), net.slayer)
# print(cmpn)

error = snn.loss(cs.net_params).to(cmpn.device)

s1 = sfi.ff.FaultSite('SF1', (slice(None), 9, 0, 0))
s2 = sfi.ff.FaultSite('SF1', (slice(None), 1, 0, 0))
s3 = sfi.ff.FaultSite('SC3', (slice(None), 2, 2, 0))
f1 = sfi.ff.Fault(sfi.fm.DeadNeuron(), s1)
f2 = sfi.ff.Fault(sfi.fm.DeadNeuron(), s2)
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

# cmpn.eject(round_idx=2)
# print(cmpn.rounds)

# cmpn.eject()
# print(cmpn.rounds)

print(cmpn)
cmpn.run(cs.test_loader, error)
# cmpn.run_complete(test_loader, SaturatedSynapse(21.), ['SF1'])

data = cmpn.export()
data.save()
cmpn.save()

data_ = sfi.CampaignData.load()
cmpn_ = data_.build()
cmpn__ = sfi.Campaign.load()
