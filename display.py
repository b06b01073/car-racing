from DQN.agent import *
import gym
from gym.wrappers import FrameStack, RecordVideo


env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
env = FrameStack(env, num_stack=4)
env = RecordVideo(env, 'videos')
agent = get_agent('duel', env.action_space.n, env.observation_space)
agent.eval_network = torch.load('DQN/model/agent_params_2000.pth', map_location='cuda')
agent.eps = 0

episodes = 10
obs, _ = env.reset()
total_reward = 0
processed_obs = agent.preprocess(obs)
step = 0
while True:
    # env.render()
    action = agent.step(processed_obs)
    step += 1
    # print(action)

    next_obs, reward, terminated, _, _ = env.step(action)

    total_reward += reward
    # print(reward, action)

    processed_next_obs = agent.preprocess(next_obs)

    processed_obs = processed_next_obs


    if terminated:
        # print('terminated by ', terminated)
        break