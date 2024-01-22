from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
import torch
from torch.utils.data import DataLoader

import slayerSNN as snn


# Configuration parameters (modify depending on application)
case_study = 'nmnist-lenet'
to_train = True
do_enable = False
epochs = 100

# To work on a new case study:
#   - Add its name in the list
#   - Make a new import case

assert case_study in ['nmnist-deep', 'nmnist-lenet', 'gesture'], \
    f'Case study {case_study} value is not valid. Please import it.'

# Case study imports
if 'nmnist' in case_study:
    yaml_file = 'nmnist'
    from models.nmnist import NMNISTDataset as CSDataset

    if case_study == 'nmnist-deep':
        from models.nmnist import NMNISTNetwork as CSNetwork
    elif case_study == 'nmnist-lenet':
        from models.nmnist import LeNetNetwork as CSNetwork

# No modifications needed after this line
# ---------------------------------------

# Generalized network/dataset initialization
device = torch.device('cuda')
net_params = snn.params(f'models/config/{yaml_file}.yaml')
net = CSNetwork(net_params).to(device)

error = snn.loss(net_params).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, amsgrad=True)
stats = snn.utils.stats()

trainingSet = CSDataset(
    data_path=net_params['training']['path']['dir_train'],
    samples_file=net_params['training']['path']['list_train'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

testingSet = CSDataset(
    data_path=net_params['training']['path']['dir_test'],
    samples_file=net_params['training']['path']['list_test'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

# Testing (without re.training)
if not to_train:
    net = torch.load(f"out/net/{case_study}{'-do' if do_enable else ''}_net.pt")

    for i, (input, target, label) in enumerate(testLoader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input, do_enable)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(0, i)

    stats.update()

    exit(0)

# Training
print(case_study + ":")
os.makedirs('out/net', exist_ok=True)

for epoch in range(epochs):
    tSt = datetime.now()

    for i, (input, target, label) in enumerate(trainLoader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input, do_enable)

        stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.training.numSamples += len(label)

        loss = error.numSpikes(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        stats.training.lossSum += loss.cpu().data.item()
        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    # Testing
    for i, (input, target, label) in enumerate(testLoader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input, do_enable)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(epoch, i)

        # Save trained network (based on the best testing accuracy)
        if stats.testing.bestAccuracy:
            torch.save(net, f"out/net/{case_study}{'-do' if do_enable else ''}_net.pt")

    stats.update()

# Save statistics
with open(f"out/net/{case_study}{'-do' if do_enable else ''}_stats.pkl", 'wb') as stats_file:
    pickle.dump(stats, stats_file)

# Plot and save the training results
plt.figure()
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing .accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'out/net/{case_study}_train.png')
