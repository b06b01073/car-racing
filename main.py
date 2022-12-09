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

# Average reward of random policy: -55.70347185719232, std: 4.753479622453348

def main(algo):
    env = gym.make('CarRacing-v2', continuous=False)
    env = FrameStack(env, num_stack=4)

    buffer_capacity = 100000
    replay_buffer = ReplayBuffer(buffer_capacity)

    episodes = 3000
    update_interval = 10
    print(env.action_space.n)
    agent = Agent.get_agent(algo, env.action_space.n, env.observation_space)    
    total_rewards = []
    check_points = [x for x in range(0, episodes + 1, 50)]
    step = 0
    losses = []

    start_time = time.time()
    for i in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        episode_loss = 0


        # obs = ignore_frames(env, frames=30)

        processed_obs = agent.preprocess(obs)

        save_image(processed_obs[0] / 255, 'image/first_game_screen.png')


        game_step = 0
        while True:
            action = agent.step(processed_obs)
            # print(action)
            game_step += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            # print(reward)

            processed_next_obs = agent.preprocess(next_obs)

            replay_buffer.append([processed_obs, action, reward, processed_next_obs, terminated])

            processed_obs = processed_next_obs

            if agent.batch_size <= len(replay_buffer):
                episode_loss += agent.learn(replay_buffer)
                

            step += 1
            if step % 3000 == 0:
                agent.eps_scheduler(mode='linear')

            if terminated or truncated:
                break
            

        if i % update_interval == 0:
            agent.hard_update()
        agent.lr_scheduler.step()

        with torch.no_grad():
            losses.append(torch.sum(torch.abs(episode_loss)).item() / game_step)
        total_rewards.append(total_reward)

        cur_time = time.time()
        time_left = str(datetime.timedelta(seconds=(cur_time - start_time) / (i + 1) * (episodes - (i + 1))))
        print(f'Episode: {i + 1}, Total reward: {total_reward},  Eps: {agent.eps}, Time Left: {time_left}')


        if i in check_points:
            torch.save(agent.eval_network, f'DQN/model/agent_params_{i}.pth')


        plt.plot(total_rewards)
        # plt.plot(avg_rewards)
        plt.savefig('rewards')
        plt.close()     

        plt.plot(losses)
        plt.savefig('losses')
        plt.close()


def ignore_frames(env, frames=30):
    for _ in range(frames):
        obs, _, _, _, _ = env.step(3)
    return obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '-a', default='DQN')

    args = parser.parse_args()
    algo = args.algo

    main(algo)