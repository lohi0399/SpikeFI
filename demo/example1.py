import os
import torch
import slayerSNN as snn
from spikefi.models import DeadNeuron, ParametricNeuron, SaturatedSynapse, BitflippedSynapse
from spikefi.fault import FaultSite, Fault
from spikefi.core import Campaign
from spikefi import visual
import demo as cs
from demo import test_loader

# Initialization
fnetname = cs.get_fnetname(trial='2')
net: cs.Network = torch.load(os.path.join(cs.OUT_DIR, cs.CASE_STUDY, fnetname))
net.eval()


# ✅ Load test dataset
test_loader = cs.test_loader
print(f"Test set size: {len(test_loader.dataset)} images")

_, spikes, targets, labels = next(iter(test_loader))

print(f"Spikes shape: {spikes.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Labels shape: {labels.shape}")


# Testing

# ✅ Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

stats = snn.utils.stats()
spike_loss = snn.loss(cs.net_params).to(device)

for i, (_, input, target, label) in enumerate(cs.test_loader, 0):
    input = input.to(device)
    target = target.to(device)

    output = net.forward(input)

    stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
    stats.testing.numSamples += len(label)

    loss = spike_loss.numSpikes(output, target)
    stats.testing.lossSum += loss.cpu().data.item()
    stats.print(0, i)