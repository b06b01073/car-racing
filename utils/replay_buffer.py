import collections
import random
import torch
from torchvision.utils import save_image

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, item):
        # the items in the buffer is of the form ([obs, action, reward, next_obs, terminated])
        self.buffer.append(item)

    def sample(self, batch_size):
        # batch is of the form (batch_size, [obs, action, reward, next_obs, terminated])
        # each item in the batch is of the form ([obs, action, reward, next_obs, terminated])
        batch = random.sample(self.buffer, batch_size)

        obs = torch.stack([experience[0] for experience in batch])
        actions = torch.Tensor([experience[1] for experience in batch]).long()
        rewards = torch.Tensor([experience[2] for experience in batch])
        next_obs = torch.stack([experience[3] for experience in batch])
        terminated = torch.Tensor([experience[4] for experience in batch]).long()

        return obs, actions, rewards, next_obs, terminated

    
    def __len__(self):
        return len(self.buffer)
        