import os
import torch

import slayerSNN as snn

import spikefi as sfi
import demo as cs


CMPN_SEL = 2

fnetname = cs.get_fnetname()
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()

s1 = sfi.ff.FaultSite(layer_name='SF1')
s2 = sfi.ff.FaultSite(layer_name='SF2')
s3 = sfi.ff.FaultSite('SC2', (slice(None), 2, 2, 0))
f1 = sfi.ff.Fault(sfi.fm.DeadNeuron(), s1)
f2 = sfi.ff.Fault(sfi.fm.SaturatedNeuron(), s2)
f3 = sfi.ff.Fault(sfi.fm.DeadNeuron(), s3)
f4 = sfi.ff.Fault(sfi.fm.SaturatedSynapse(2.), random_sites_num=1)
f5 = sfi.ff.Fault(sfi.fm.DeadNeuron(), random_sites_num=2)
f6 = sfi.ff.Fault(sfi.fm.ParametricNeuron('theta', 0.5), [sfi.ff.FaultSite('SC1'), sfi.ff.FaultSite('SC1')])
f7 = sfi.ff.Fault(sfi.fm.ParametricNeuron('tauSr', 0.1), sfi.ff.FaultSite('SC2'))
f8 = sfi.ff.Fault(sfi.fm.SaturatedNeuron(), [sfi.ff.FaultSite('SC2')])

if CMPN_SEL == 1:
    cmpn1 = sfi.Campaign(net, cs.shape_in, net.slayer,
                         name=fnetname.removesuffix('.pt') + '_demo1')

    cmpn1.inject([f4])
    cmpn1.inject([f1, f2, f3])
    cmpn1.then_inject([f5, f6])
    cmpn1.then_inject([f7])
    cmpn1.then_inject([f8])

    cmpn1.eject(round_idx=1)

    print(cmpn1)
    cmpn1.run(cs.test_loader, error=snn.loss(cs.net_params).to(cmpn1.device),
              opt=sfi.CampaignOptimization.O4)

    cmpn1.save()
    print(f"{cmpn1.duration : .2f} secs")

elif CMPN_SEL == 2:
    cmpn2 = sfi.Campaign(net, cs.shape_in, net.slayer,
                         name=fnetname.removesuffix('.pt') + '_synapse_dead_SF1')

    cmpn2.inject_complete(sfi.fm.DeadSynapse(), layer_names=['SF1'])

    print(cmpn2.name)
    cmpn2.run(cs.test_loader, error=snn.loss(cs.net_params).to(cmpn2.device))

    cmpn2.save()
    print(f"{cmpn2.duration : .2f} secs")
