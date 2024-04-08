import os

import spikefi as sfi
import demo as cs


layer_name = 'SF2'

fnetname = cs.get_fnetname(trial='4')
fres_path = os.path.join(cs.CASE_STUDY, fnetname.removesuffix('.pt') + "_synapse_bitflip_SF2.pkl")

cmpn_data = sfi.CampaignData.load(os.path.join(sfi.utils.io.RES_DIR, fres_path))
print(cmpn_data.name)

sfi.visual.heat(cmpn_data, format='png')
