import spikefi as sfi
import demo as cs

data = sfi.CampaignData.load(sfi.utils.io.make_res_filepath(cs.base_fname + '_neuron_dead_sf1.pkl'))

sfi.visual.plot_neuronal(data, sfi.fm.DeadNeuron())
# sfi.visual.plot_synaptic(data, sfi.fm.SaturatedSynapse(21.))
