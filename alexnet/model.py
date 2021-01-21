import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=1) #11 with stride 4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1) #5
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1)
        #self.conv4 = nn.Conv2d(256, 512, kernel_size=1)
        #self.conv5 = nn.Conv2d(512, 256, kernel_size=1)

        self.drop = nn.Dropout(0.5)
        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.pool = nn.MaxPool2d(kernel_size=3)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.norm(self.conv1(x))))
        x = self.pool(F.relu(self.norm(self.conv2(x))))
        x = self.pool(F.relu(self.norm(self.conv3(x))))

        #x = self.pool(F.relu(self.norm(self.conv4(x))))
        #x = self.pool(F.relu(self.norm(self.conv5(x))))

        x = self.flatten(x)

        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        #output = self.fc3(x)
        output = F.sigmoid(self.fc3(x))

        return output 
