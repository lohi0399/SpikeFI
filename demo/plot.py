import spikefi as sfi

cmpn = sfi.Campaign.load('./out/nmnist_syn_satu21_SF1.pkl')
data = cmpn.export()

sfi.visual.plot_synaptic(data, sfi.fault.SaturatedSynapse(21.))
