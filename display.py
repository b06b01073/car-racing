from DQN.agent import *
import gym
import collections

def main():
    frames = 4

    # diff: 0 [1, 4, (8)]
    env = gym.make('ALE/Breakout-v5', render_mode='human', frameskip=3, repeat_action_probability=0, mode=4, difficulty=0)

    while True:
        model = torch.load('DQN/model/agent_params_100.pth')
        agent = DQNAgent(env.action_space.n, env.observation_space, train_mode=False)
        agent.eval_network.model = model
        obs, _ = env.reset()

        transition_queue = collections.deque(maxlen=frames)
        processed_obs = agent.preprocess(obs)
        transition_queue.append(processed_obs)
        stacked_obs = stack_obs(transition_queue)
        total_reward = 0
        while True:
            env.render()

            random_choice = random.choice(range(env.action_space.n))
            action = random_choice if len(transition_queue) != transition_queue.maxlen else agent.step(stacked_obs)
            print(action)


            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward


            processed_next_obs = agent.preprocess(next_obs)
            
            transition_queue.append(processed_next_obs)
            # the stacked_next_obs is assigned until the next_obs is pushed into the queue
            stacked_next_obs = stack_obs(transition_queue)

            stacked_obs = stacked_next_obs

            if terminated or truncated:
                break

        print(total_reward)

def stack_obs(queue):
    return [obs for obs in queue]

if __name__ == '__main__':
    main()