import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import random
from torch.optim import lr_scheduler
import torchvision.transforms.functional as TF
from PIL import Image
from . import hyperparams 


# this import the CNN(note that the cwd need to be the root folder)
from utils.CNN import CNN, DuelCNN

random.seed(777)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = hyperparams.config
lr = config['lr']
gamma = config['gamma']
eps = config['eps']
eps_low = config['eps_low']
eps_decay = config['eps_decay']
eps_step = config['eps_step']
tau = config['tau']
batch_size = config['batch_size']

print(lr, gamma, eps)

class Agent():
    def __init__(self, action_dim, obs_dim, train_mode=True, duel=False):
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(obs_dim)
        self.eval_network = DuelCNN(action_dim) if duel else CNN(action_dim) 
        self.target_network = DuelCNN(action_dim) if duel else CNN(action_dim)
        self.target_network.load_state_dict(self.eval_network.state_dict())


        print(f'DQNAgent: The network is on {device}')


        self.train_mode = train_mode

        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.RMSprop(self.eval_network.parameters(), lr=lr, weight_decay=0.01)

        self.gamma = gamma
        self.eps = eps
        self.eps_low = eps_low
        self.eps_decay = eps_decay
        self.eps_step = 0.005
        self.step_count = 0

        self.tau = tau
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optim, milestones=[100, 300, 500, 700, 1400, 2100], gamma=0.1)


    def preprocess(self, obs):
        # convert (h, w, c) to (c, h, w)

        transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((120, 84)),
            transforms.CenterCrop((84, 84)),
        ])
        obs = np.asarray(obs)
        stacked_obs = []

        for i in range(len(obs)):
            stacked_obs.append(transformer(torch.Tensor(obs[i]).permute(2, 0, 1)).squeeze())

        return torch.stack(stacked_obs)


    def eps_scheduler(self, mode='linear'):
        if mode == 'exp':
            self.eps *= self.eps_decay
        if mode == 'linear':
            self.eps -= self.eps_step

        self.eps = max(self.eps, self.eps_low)


    def soft_update(self):
        for target_param, eval_param in zip(self.target_network.parameters(), self.eval_network.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)


    def hard_update(self):
        self.target_network.load_state_dict(self.eval_network.state_dict())


class DQNAgent(Agent):
    def __init__(self, action_dim, obs_dim, train_mode=True, duel=False):
        super().__init__(action_dim, obs_dim, train_mode, duel=duel)

    def step(self, obs):
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(device) # obs.shape is (1, 4, h, w), the unsqueeze treat the obs as a batch with batch size = 1

            action_scores = self.eval_network(obs).squeeze()


            # return the action based on epsilon-greedy strategy
            return random.randint(0, self.action_dim - 1) if random.random() < self.eps else torch.argmax(action_scores).item()


    def learn(self, replay_buffer):
        obs, actions, rewards, next_obs, terminals = replay_buffer.sample(self.batch_size)


        Q_eval = self.eval_network(obs.to(device)).gather(1, actions.reshape(-1, 1).to(device)).flatten()
        Q_target = torch.max(self.target_network(next_obs.to(device)), dim=1)[0]
        y = rewards.to(device) + self.gamma * Q_target * (1 - terminals.to(device))

        loss = self.loss_fn(Q_eval, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step() 

        return loss


class DDQNAgent(Agent):
    def __init__(self, action_dim, obs_dim, train_mode=True):
        super().__init__(action_dim, obs_dim, train_mode)

    def step(self, stacked_obs):
        # the concat_experiences is of the form (frames, [obs, action, reward, next_obs]), we need only the (4, obs) to make the decision 
        with torch.no_grad():
            obs = torch.stack(stacked_obs).unsqueeze(0).to(device) 

            action_scores = self.eval_network(obs)
            if not self.train_mode:
                return torch.argmax(action_scores).item()

            # return the action based on epsilon-greedy strategy
            
            return random.randint(0, self.action_dim - 1) if random.random() < self.eps else torch.argmax(action_scores).item()


    def learn(self, replay_buffer):
        obs, actions, rewards, next_obs, terminated = replay_buffer.sample(self.batch_size)

        obs = obs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_obs = next_obs.to(device)
        terminated = terminated.to(device)

        Q_eval = self.eval_network(obs).gather(1, actions.reshape(-1, 1)).squeeze()

        policy_actions = torch.max(self.eval_network.forward(obs), dim=1)[1]

        Q_target = self.target_network.forward(next_obs).gather(1, policy_actions.reshape(-1, 1)).squeeze()

        y = rewards + self.gamma * Q_target * (1 - terminated)

        loss = self.loss_fn(Q_eval, y)
        self.optim.zero_grad()
        loss.backward()

        for param in self.eval_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optim.step() 

def get_agent(algo, action_dim, obs_dim):
    if algo == 'DQN':
        return DQNAgent(action_dim, obs_dim)
    if algo == 'DDQN':
        return DDQNAgent(action_dim, obs_dim)
    if algo == 'duel':
        return DQNAgent(action_dim, obs_dim, duel=True)

