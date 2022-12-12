from DQN.agent import *
import gym
from gym.wrappers import FrameStack
from tqdm import tqdm


# candidates model are from episoe 800 to 1950
model_paths = [f'DQN/model/duel/agent_params_{i}.pth' for i in range(800, 1951, 50)]


env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
env = FrameStack(env, num_stack=4)
agent = get_agent('duel', env.action_space.n, env.observation_space)

attempts = 50
best_reward = float('-inf')
best_model = None

for model_path in model_paths:
    print(f'{model_path} is playing.')
    agent.eval_network = torch.load(model_path, map_location='cuda')
    agent.eps = 0
    total_reward = 0

    for i in tqdm(range(attempts)):
        obs, _ = env.reset()
        processed_obs = agent.preprocess(obs)
        while True:
            # env.render()
            with torch.no_grad():
                action = agent.step(processed_obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            processed_next_obs = agent.preprocess(next_obs)
            processed_obs = processed_next_obs

            if terminated or truncated:
                break
            
    print(f'total_reward: {total_reward}, avg_reward: {total_reward / attempts}')

    if total_reward > best_reward:
        best_model = model_path
        best_reward = total_reward
        print(f'The best model is now {model_path}.')
    

print(f'The best model is {model_path}')
