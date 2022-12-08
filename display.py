from DQN.agent import *
import gym
from gym.wrappers import FrameStack, RecordVideo


env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
env = RecordVideo(env, 'videos')
env = FrameStack(env, num_stack=4)
agent = get_agent('DQN', env.action_space.n, env.observation_space)

model = torch.load('DQN/model/agent_params_100.pth')
agent.eval_network.model = model
agent.eps = 0
episodes = 10
for i in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    episode_loss = 0

    for _ in range(50):
        _, _, _, _, _ = env.step(0)

    obs, _, _, _, _ = env.step(0)
    processed_obs = agent.preprocess(obs)

    while True:
        action = agent.step(processed_obs)
        # print(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        print(reward, action)

        processed_next_obs = agent.preprocess(next_obs)

        processed_obs = processed_next_obs


        if terminated or truncated:
            print('terminated by ', terminated, truncated)
            break
        