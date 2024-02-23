import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    """Simple convolutional model for cifar10 dataset."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        # model = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(128 * 4 * 4, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 40),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 10),
        #     torch.nn.Softmax(dim=1)
        # )
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.final = nn.Linear(128, 10)

    def forward(self, x):
        # step by step forward pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.final(x)
        x = F.softmax(x, dim=1)
        return x