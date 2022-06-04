import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Convolutional neural network of specified architecture
class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(16, 2, 5, padding=2)
        # Set Dropout to desired percentage
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.upsample(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.conv4(x)
        return x