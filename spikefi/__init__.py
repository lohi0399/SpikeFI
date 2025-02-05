# This line (__all__) ensures only these modules and objects are imported when someone does 'from spikefi import *' while hiding the others
__all__ = ["core", "fault", "ff", "models", "fm", "visual",
           "utils",
           "Campaign", "CampaignData", "CampaignOptimization"]

from spikefi.core import Campaign, CampaignData, CampaignOptimization
from spikefi import fault as ff
from spikefi import models as fm
from spikefi import visual
from spikefi import utils

import os

os.makedirs(utils.io.FIG_DIR, exist_ok=True)
os.makedirs(utils.io.RES_DIR, exist_ok=True)
