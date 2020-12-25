import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

data_dir = "./frame_data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.ImageFolder(
    root = data_dir,
    transform = transform
)
num_data = len(dataset)

train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8 * num_data), num_data - int(0.8 * num_data)])

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size = 4,
    shuffle=True,
    num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size = 4,
    shuffle=False,
    num_workers=2
)

classes = ('D', 'Dp', 'R', 'Rp', 'U', 'Up', 'N')

import torch.nn as nn
import torch.nn.functional as F

# Architecture is completely made up right now, FIX.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 1, 7)
        self.avgpool = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(31284, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.avgpool(F.relu(self.conv2(x)))
        x = x.view(-1, 31284)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 30 == 29:    # print every 30 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 30))
            running_loss = 0.0

print('Finished Training')