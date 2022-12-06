import gym
import DQN.agent as Agent
from utils.replay_buffer import ReplayBuffer
import torch
import matplotlib.pyplot as plt
import time
import datetime
from torchvision.utils import save_image
import argparse
from gym.wrappers import FrameStack

def main(algo):
    start_time = time.time()
    env = gym.make('CarRacing-v2', continuous=False)
    env = FrameStack(env, num_stack=4)

    buffer_capacity = 100000
    replay_buffer = ReplayBuffer(buffer_capacity)

    episodes = 1500
    update_interval = 10
    print(env.action_space.n)
    agent = Agent.get_agent(algo, env.action_space.n, env.observation_space)    
    total_rewards = []
    check_points = [x for x in range(0, episodes + 1, 100)]
    step = 0

    for i in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        processed_obs = agent.preprocess(obs)

        while True:
            action = agent.step(processed_obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
            processed_next_obs = agent.preprocess(next_obs)

            replay_buffer.append([processed_obs, action, reward, processed_next_obs, terminated or truncated])

            processed_obs = processed_next_obs

            if agent.batch_size <= len(replay_buffer):
                agent.learn(replay_buffer)

            step += 1
            if step % 400 == 0:
                agent.eps_scheduler(mode='linear')

            if terminated or truncated:
                break
            

        if i % update_interval == 0:
            # agent.hard_update()
            agent.soft_update()

        total_rewards.append(total_reward)

        cur_time = time.time()
        time_left = str(datetime.timedelta(seconds=(cur_time - start_time) / (i + 1) * (episodes - (i + 1))))
        print(f'Episode: {i + 1}, Total reward: {total_reward},  Eps: {agent.eps}, Time Left: {time_left}')


        if i in check_points:
            torch.save(agent.eval_network, f'DQN/model/agent_params_{i}.pth')


        plt.plot(total_rewards)
        # plt.plot(avg_rewards)
        plt.savefig('car racing')
        plt.close()     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '-a', default='DQN')

    args = parser.parse_args()
    algo = args.algo

    main(algo)