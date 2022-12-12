# Car Racing

## Introduction

This is a side project implemented by me with some insightful advices from [wasd9813](https://github.com/wasd9813). In this project, I implemented the dueling DQN to solve the [car racing problem](https://www.gymlibrary.dev/environments/box2d/car_racing/#car-racing). Unlike the project I did before [here](https://github.com/b06b01073/classic-control), the agent can only observe raw pixels of the game screen, which makes the task more challenging. The well-trained agent should have the ability to extract the information from raw pixels, and make good decisions so that the race car will stay on the track instead of running out of it. After training the agent for thousands of episodes, it learned to run on the track, make turns and recover from mistakes(such as running out of the track accidentally). The video recording of the result is posted below.


https://user-images.githubusercontent.com/56951221/207053746-de6b0daa-d1b8-43d9-9c2f-40e4f4550982.mp4


Note: Details of the environment such as reward, observation space and action space is in the link [here](https://www.gymlibrary.dev/environments/box2d/car_racing/#car-racing). 

## Model Architecture
The game screen is first transformed to grayscale image and cropped in the center, then it is stacked with the next 3 frames of the gameplay which goes through the same process to form a single observation record, this preprocessing method is similar to the method proposed by DeepMind [here](https://arxiv.org/pdf/1312.5602.pdf). The observation is then passed to the dueling DQN, which is compose of convolutional layers and fully connected layers. The visualization of the duel DQN model architecture is down below.

![Model architecture](https://github.com/b06b01073/atari/blob/master/image/model.png)

## Parameters of the Model

### Hyperparameters
The hyperparameters of the model can be found in the `DQN/hyperparams.py` file. The exploration rate is set to 1 at the beginning and decays for every 3000 frames(`eps_decay_interval`) of gameplay with the step size of 0.005, the minimum value of the exploraion rate is set to 0.1 during the experiment.

### Model Checkpoints
The process save the model for every 50 episodes of training, this can be modified in `main.py`.


## How to Execute the Process?
```
$python main.py             # to run the DQN(the default algorithm)
$python main.py --algo duel # to run the dueling DQN
```
You can also record the video by
```
$python display.py [--algo] [--model]
```
for example,
```
$python display.py --algo duel --model 300
```
will create an agent with dueling DQN, and the parameters of the network is from `DQN/model/duel/agent_params_300.pth`.


Since an episode is truncated automatically when it reach the 1000th frame of the gameplay(this mechanism prevents a stationary agent that makes the entire process being executed indefinitely), the `RecordVideo` wrapper can only record at most 20s of the gameplay.

## Evaluation and Result

### Policy Comparison
The average reward of random policy is roughly -55.7034(calculated from 100 episodes of random gameplay). The average reward of a human player is 802.745(from 20 episodes of gameplay, the data can be found in `result\human\reward.txt`). The best trained agent obtained an average reward of 742.591(from 20 episodes of gameplay, the data can be found in `result\human\reward.txt`), which reach human-level gameplay(the best agent is selected by `eval.py`).

|  player   | average reward  |
|  ----  | ----  |
| random policy | -55 |
| human  | 802.745 |
| DRL agent  | 742.591 |

### Dueling DQN Result
![Dueling DQN reward](https://github.com/b06b01073/atari/blob/master/result/duel/reward.png)
![eps](https://github.com/b06b01073/atari/blob/master/result/duel/eps.png)

### DQN Result
![DQN reward](https://github.com/b06b01073/atari/blob/master/result/DQN/reward.png)
![eps](https://github.com/b06b01073/atari/blob/master/result/DQN/eps.png)

## References
1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf?source=post_page)
2. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
