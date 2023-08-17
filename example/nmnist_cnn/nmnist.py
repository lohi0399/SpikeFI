from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import os
import pickle

# TODO: Rename variables/classes according to naming rules and remove unecessary code

example_dir = os.path.dirname(__file__)
net_params = snn.params(os.path.join(example_dir, 'nmnist.yaml'))
test_only = True


# Dataset definition
class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]

        inputSpikes = snn.io.read2Dspikes(
                        self.path + str(inputIndex.item()) + '.bs2'
                        ).toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)),
                                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


# Network definition
class nmnistNetwork(torch.nn.Module):
    def __init__(self, net_params):
        super(nmnistNetwork, self).__init__()
        # initialize slayer
        slayer = snn.layer(net_params['neuron'], net_params['simulation'])
        self.slayer = slayer
        # define network functions
        self.SC1 = slayer.conv(2, 16, 5, padding=1)
        self.SC2 = slayer.conv(16, 32, 3, padding=1)
        self.SC3 = slayer.conv(32, 64, 3, padding=1)
        self.SP1 = slayer.pool(2)
        self.SP2 = slayer.pool(2)
        self.SF1 = slayer.dense((8, 8, 64), 10)

    def forward(self, s_in):
        s_out1 = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))    # 32, 32, 16
        s_out2 = self.slayer.spike(self.slayer.psp(self.SP1(s_out1)))  # 16, 16, 16
        s_out3 = self.slayer.spike(self.slayer.psp(self.SC2(s_out2)))  # 16, 16, 32
        s_out4 = self.slayer.spike(self.slayer.psp(self.SP2(s_out3)))  #  8,  8, 32
        s_out5 = self.slayer.spike(self.slayer.psp(self.SC3(s_out4)))  #  8,  8, 64
        s_out  = self.slayer.spike(self.slayer.psp(self.SF1(s_out5)))  # 10

        return s_out


if __name__ == '__main__':
    # # Extract NMNIST samples
    # import os
    # with zipfile.ZipFile('NMNISTsmall.zip') as zip_file:
    #     for member in zip_file.namelist():
    #         if not os.path.exists('./' + member):
    #             zip_file.extract(member, './')

    # Define the cuda device to run the code on.
    device = torch.device('cuda')
    # Use multiple GPU's if available
    # device = torch.device('cuda:2') # should be the first GPU of deviceIDs
    # deviceIds = [2, 3, 1]

    # Create network instance.
    net = nmnistNetwork(net_params).to(device)
    # Split the network to run over multiple GPUs
    # net = torch.nn.DataParallel(Network(net_params).to(device), device_ids=deviceIds)

    # Create snn loss instance.
    error = snn.loss(net_params).to(device)

    # Define optimizer module.
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, amsgrad=True)

    # Dataset and dataLoader instances.
    trainingSet = nmnistDataset(
        datasetPath=os.path.join(example_dir, '../..', net_params['training']['path']['dir_train']),
        sampleFile=os.path.join(example_dir, '../..', net_params['training']['path']['list_train']),
        samplingTime=net_params['simulation']['Ts'],
        sampleLength=net_params['simulation']['tSample'])
    trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

    testingSet = nmnistDataset(
        datasetPath=os.path.join(example_dir, '../..', net_params['training']['path']['dir_test']),
        sampleFile=os.path.join(example_dir, '../..', net_params['training']['path']['list_test']),
        samplingTime=net_params['simulation']['Ts'],
        sampleLength=net_params['simulation']['tSample'])
    testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

    # Learning stats instance.
    stats = snn.utils.stats()

    # # Visualize the network.
    # for i in range(5):
    #   input, target, label = trainingSet[i]
    #   snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 34, 34, -1)).cpu().data.numpy()))

    if test_only:
        net = torch.load(os.path.join(example_dir, 'out/nmnist.pt'))

        # Testing loop.
        for i, (input, target, label) in enumerate(testLoader, 0):
            input = input.to(device)
            target = target.to(device)

            output = net.forward(input)

            stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.print(0, i)

        # Update stats.
        stats.update()

        exit()

    # training loop
    for epoch in range(100):
        tSt = datetime.now()

        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            # Move the input and target to correct GPU.
            input = input.to(device)
            target = target.to(device)

            # Forward pass of the network.
            output = net.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            # Reset gradients to zero.
            optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
        for i, (input, target, label) in enumerate(testLoader, 0):
            input = input.to(device)
            target = target.to(device)

            output = net.forward(input)

            stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.print(epoch, i)

        # Update stats.
        stats.update()

    # Save trained network.
    os.makedirs(os.path.join(example_dir, 'out'), exist_ok=True)
    torch.save(net, os.path.join(example_dir, 'out/nmnist.pt'))

    # Save statistics
    with open(os.path.join(example_dir, 'out/nmnist.pkl'), 'wb') as stats_file:
        pickle.dump(stats, stats_file)

    # Plot the results.
    plt.figure(1)
    plt.semilogy(stats.training.lossLog, label='Training')
    plt.semilogy(stats.testing .lossLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label='Training')
    plt.plot(stats.testing .accuracyLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
