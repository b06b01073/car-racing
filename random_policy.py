import gym
import random
import statistics
from tqdm import tqdm

random.seed(301)

def main():
    env = gym.make('CarRacing-v2', continuous=False)
    episodes = 100
    total_rewards = []


    for _ in tqdm(range(episodes)):
        env.reset()
        total_reward = 0
        while True:
            action = random.randint(0, env.action_space.n - 1)
            obs, reward, terminated, truncated, _ = env.step(action) 
            total_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(total_reward)

    print(f'Average reward of random policy: {statistics.mean(total_rewards)}, std: {statistics.stdev(total_rewards)}')

if __name__ == '__main__':
    main()