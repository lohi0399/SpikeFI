import torch
from torch.utils.data import DataLoader

import slayerSNN as snn

import spikefi as sfi
import example as cs

net: torch.nn.Module = torch.load(f"{cs.out_dir}/{cs.case_study}{'-do' if cs.do_enable else ''}.pt")
net.eval()
net_params = snn.params(f'example/config/{cs.fyamlname}.yaml')

testing_set = cs.CSDataset(
    datasetPath=net_params['training']['path']['dir_test'],
    sampleFile=net_params['training']['path']['list_test'],
    samplingTime=net_params['simulation']['Ts'],
    sampleLength=net_params['simulation']['tSample'])
test_loader = DataLoader(dataset=testing_set, batch_size=12, shuffle=False, num_workers=4)

cmpn = sfi.Campaign(net, (2, 34, 34), net.slayer)
# print(cmpn)

error = snn.loss(net_params).to(cmpn.device)

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
cmpn.run(test_loader, error)
# cmpn.run_complete(test_loader, SaturatedSynapse(21.), ['SF1'])

data = cmpn.export()
cmpn.save()

cmpn_ = sfi.Campaign.from_object(data)
cmpn__ = sfi.Campaign.load()
