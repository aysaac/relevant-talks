import torch.nn as nn
from torch.nn.functional import relu
from torch import flatten


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 5, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            5, 16, kernel_size=(5, 5), stride=(2, 2), padding=1
        )
        self.lin1 = nn.Linear(144, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.15)

    # Defining the forward pass
    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = self.pool(x)
        x = flatten(x, 1)  # flatten all dimensions except batch
        x = relu(self.lin1(x))
        x = self.drop1(x)
        x = relu(self.lin2(x))
        x = self.drop2(x)
        x = self.lin3(x)
        return x
