from DQN.agent import *
import gym
from gym.wrappers import FrameStack, RecordVideo


env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
env = FrameStack(env, num_stack=4)
env = RecordVideo(env, 'videos')
agent = get_agent('DQN', env.action_space.n, env.observation_space)
agent.eval_network = torch.load('DQN/model/agent_params_500.pth', map_location='cuda')
agent.eps = 0

episodes = 10
for i in range(episodes):
    obs, _ = env.reset()
    total_reward = 0


    processed_obs = agent.preprocess(obs)
    while True:
        action = agent.step(processed_obs)
        # print(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        # print(reward, action)

        processed_next_obs = agent.preprocess(next_obs)

        processed_obs = processed_next_obs


        if terminated or truncated:
            print('terminated by ', terminated, truncated)
            break
    print(reward)