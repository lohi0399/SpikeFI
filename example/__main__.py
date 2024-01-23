from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
import torch
from torch.utils.data import DataLoader

import slayerSNN as snn

import example as cs

# Generalized network/dataset initialization
device = torch.device('cuda')
net_params = snn.params(f'example/config/{cs.fyamlname}.yaml')
net = cs.CSNetwork(net_params).to(device)

error = snn.loss(net_params).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, amsgrad=True)
stats = snn.utils.stats()

trainingSet = cs.CSDataset(
    data_path=net_params['training']['path']['dir_train'],
    samples_file=net_params['training']['path']['list_train'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

testingSet = cs.CSDataset(
    data_path=net_params['training']['path']['dir_test'],
    samples_file=net_params['training']['path']['list_test'],
    sampling_time=net_params['simulation']['Ts'],
    sample_length=net_params['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

base_fname = f"{cs.case_study}{'-do' if cs.do_enable else ''}"
trial = str(len([f for f in os.listdir(cs.out_dir) if base_fname in f and f.endswith('.pt')]) or '')
fnetname = f"{base_fname}_net{trial}.pt"
fstaname = f"{base_fname}_stats{trial}.pkl"
ffigname = f"{base_fname}_train{trial}.svg"

# Testing (without re.training)
if not cs.to_train:
    net = torch.load(f"out/net/{cs.case_study}{'-do' if cs.do_enable else ''}_net.pt")

    for i, (input, target, label) in enumerate(testLoader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input, cs.do_enable)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(0, i)

    stats.update()

    exit(0)

# Training
print(cs.case_study + ":")
os.makedirs('out/net', exist_ok=True)

for epoch in range(cs.epochs):
    tSt = datetime.now()

    for i, (input, target, label) in enumerate(trainLoader, 0):
        input = input.to(device)
        target = target.to(device)

        output = net.forward(input, cs.do_enable)

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

        output = net.forward(input, cs.do_enable)

        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
        stats.testing.numSamples += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(epoch, i)

    stats.update()

    # Save trained network (based on the best testing accuracy)
    if stats.testing.accuracyLog[-1] == stats.testing.maxAccuracy:
        torch.save(net, os.path.join(cs.out_dir, fnetname))

# Save statistics
with open(fstaname, 'wb') as stats_file:
    pickle.dump(stats, stats_file)

# Plot and save the training results
plt.figure()
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing .accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(ffigname)
