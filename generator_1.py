import torch
import torch.nn as nn

# Generator class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),  # 100 random numbers as input
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),  # MNIST image size is 28*28 = 784
            nn.Tanh()  # Output range from -1 to 1
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)  # Reshape the output to be a 28x28 imag
    