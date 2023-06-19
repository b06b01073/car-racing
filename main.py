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
import collections
from statistics import mean

# Average reward of random policy: -55.70347185719232, std: 4.753479622453348

def main(algo):
    env = gym.make('CarRacing-v2', continuous=False)
    env = FrameStack(env, num_stack=4)
    print(env.action_space.n)

    buffer_capacity = 100000
    replay_buffer = ReplayBuffer(buffer_capacity)

    episodes = 2000
    update_interval = 10
    eps_decay_interval = 1000


    agent = Agent.get_agent(algo, env.action_space.n, env.observation_space)    
    total_rewards = []
    eps_history = []
    check_points = [x for x in range(0, episodes + 1, 50)]
    reward_window = collections.deque(maxlen=100)
    reward_means = []
    step = 0

    start_time = time.time()
    for i in range(episodes):
        obs = env.reset()
        total_reward = 0
        episode_loss = 0
        eps_history.append(agent.eps)

        processed_obs = agent.preprocess(obs)

        total_loss = 0

        while True:
            action = agent.step(processed_obs)

            next_obs, reward, terminated,  _ = env.step(action)

            total_reward += reward

            processed_next_obs = agent.preprocess(next_obs)

            replay_buffer.append([processed_obs, action, reward, processed_next_obs, terminated])

            processed_obs = processed_next_obs

            if agent.batch_size <= len(replay_buffer):
                total_loss += agent.learn(replay_buffer)
                

            step += 1
            if step % eps_decay_interval == 0:
                agent.eps_scheduler(mode='linear')

            if terminated:
                break
            

       
        agent.lr_scheduler.step()

        total_rewards.append(total_reward)

        cur_time = time.time()
        time_left = str(datetime.timedelta(seconds=(cur_time - start_time) / (i + 1) * (episodes - (i + 1))))
        print(f'Episode: {i + 1}, Total reward: {total_reward},  Eps: {agent.eps}, Time Left: {time_left}, Total Loss: {total_loss}')


        if i in check_points:
            torch.save(agent.eval_network, f'DQN/model/{algo}/agent_params_{i}.pth')

        reward_window.append(total_reward)
        reward_means.append(mean(reward_window))


        # plot the result
        plt.plot(total_rewards, label='Total Reward')
        # plt.plot(avg_rewards)
        plt.plot(reward_means, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
        plt.title(algo)

        plt.savefig(f'result/{algo}/reward')
        plt.close()     

        plt.plot(eps_history)
        plt.title('Exploration Rate')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.savefig(f'result/{algo}/eps')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '-a', default='DQN')

    args = parser.parse_args()
    algo = args.algo

    main(algo)