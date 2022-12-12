from DQN.agent import *
import gym
from gym.wrappers import FrameStack, RecordVideo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=int, default=0)
parser.add_argument('--algo', '-a', default='DQN')
args = parser.parse_args()
model = args.model
algo = args.algo

model_path = f'DQN/model/{algo}/agent_params_{args.model}.pth'




env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
env = FrameStack(env, num_stack=4)
env = RecordVideo(env, 'videos')
agent = get_agent(algo, env.action_space.n, env.observation_space)
agent.eval_network = torch.load(model_path, map_location='cuda')
agent.eps = 0

obs, _ = env.reset()
total_reward = 0
processed_obs = agent.preprocess(obs)
while True:
    # env.render()
    with torch.no_grad():
        action = agent.step(processed_obs)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    processed_next_obs = agent.preprocess(next_obs)
    processed_obs = processed_next_obs

    if terminated or truncated:
        # print('terminated by ', terminated)
        break