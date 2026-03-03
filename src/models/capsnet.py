import torch
import torch.nn as nn

class SimpleCapsNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCapsNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, kernel_size=9)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(256 * 56 * 56, num_classes)  # adjust for image size

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x