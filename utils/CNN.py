import torch.nn as nn
import torch
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ).to(device)

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x.view(x.shape[0], -1))

        return x


class DuelCNN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        print('Using dueling dqn')

        self.action_dim = action_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ).to(device)

        self.state_value = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        self.advantage = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        ).to(device)


    def forward(self, x):        
        y = self.cnn(x)
        y = y.view(y.shape[0], -1)

        state_value = self.state_value(y)
        adavantage = self.advantage(y)

        return state_value + (adavantage - adavantage.mean())

