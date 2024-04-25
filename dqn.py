import numpy as np
import gymnasium as gym
import random
from datetime import datetime
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import namedtuple, deque
import os
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using \"{DEVICE}\" device.")

# Render_mode set to rgb_array to allow dqn network to run on images
env = gym.make("FrozenLake-v1", render_mode='human')
env.reset()

# Use a global plot to support live rendering
# FIG, AX = plt.subplots()
# IMG = AX.imshow(env.render())
# plt.show(block=False)


def render_env(env, title=None):
    """
    Render environment image.

    :param env: Frozenlake environment

    :return: None
    """

    # if title:
    #     FIG.suptitle(title)
    #
    # IMG.set_data(env.render())
    # FIG.canvas.draw()
    # plt.pause(0.0001)


def render_state(env, state_idx, transforms):
    """
    Render state image.

    :param env: Frozenlake environment
    :param state_idx: Index of the state, range is 0-15
    :param transforms: Transform the image before return

    :return: None
    """
    image = extract_state_img(env, state_idx, transforms).permute(1, 2, 0)
    plt.imshow(image, cmap='gray')


def extract_state_img(env, state_idx, transforms):
    """
    Extracts the state image from the environment image.

    :param env: Frozenlake environment
    :param state_idx: Index of the state, range is 0-15
    :param transforms: Transform the image before return

    :return: Image of shape CxHxW
    """
    # Convert env rgb array to tensor
    env = torch.tensor(env.render())
    block_size = env.shape[0] // 4

    # Extract state from given index
    env = env.permute(2, 0, 1)
    env = transforms(env)
    env = env.permute(1, 2, 0)

    start_idx = (state_idx // 4) * block_size
    end_idx = (state_idx % 4) * block_size
    state_img = env[start_idx:(start_idx + block_size + 2 * PADDING), end_idx:(end_idx + block_size + 2 * PADDING), :]

    state_img = state_img.permute(2, 0, 1).type(torch.float)
    return state_img


# Constants used throughout the code
ACTION_SPACE_SIZE = env.action_space.n
STATE_SPACE_SIZE = env.observation_space.n

PADDING = 20
TAU = 0.0005
GAMMA = 0.99
LR = 1e-4

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.Pad(padding=PADDING, fill=255),
    torchvision.transforms.Grayscale(1),
    torchvision.transforms.Lambda(lambda x: x / 255.0),
])

IMG_WIDTH = extract_state_img(env, state_idx=9, transforms=TRANSFORMS).shape[1]

"""
Replay Memory
"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        if len(self.memory) < self.batch_size:
            return None
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


"""
DQN Architecture
"""


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(9 * 9 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.model(x)


"""
DQN Agent
"""


class Agent():
    def __init__(self, env, policy_net, target_net, optimizer):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = ReplayMemory(capacity=10_000, batch_size=128)
        self.tau = TAU
        self.gamma = GAMMA
        self.checkpoint_freq = 100

    def train(self, n_episodes, n_steps, exploration_rate=1.0, resume_training=False, pretrained_model=None):
        """
        Train a model with the following params

        :param n_episodes: Total episodes to train for
        :param n_steps: Total steps or actions before each episode is terminated
        :param save_dir: Directory where model is save at each checkpoint
        :param exploration_rate: Rate between exploration and exploitation
        :param resume_training: Set to True to continue training from last checkpoint saved in save_dir

        :return: None
        """
        if resume_training:
            self.load_model(pretrained_model)

        for episode in range(n_episodes):
            # Save model every 20 episodes
            if episode > 0 and episode % self.checkpoint_freq == 0:
                self.save_model(episode)

            state, info = self.env.reset()
            print(f"Episode: {episode}")

            for step in range(n_steps):
                state_img = extract_state_img(self.env, state, transforms=TRANSFORMS).to(DEVICE)

                # Select an action via explore vs exploit
                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * episode / EPS_DECAY)
                if sample > eps_threshold or resume_training:
                    with torch.no_grad():
                        action = torch.argmax(self.policy_net(state_img.unsqueeze(dim=0))).item()
                else:
                    action = self.env.action_space.sample()

                # Execute action, observe reward, and store experience
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # If done
                done = terminated or truncated
                if terminated:
                    next_state = None

                title = f"Episode:{episode}    Step: {step}"
                render_env(self.env, title)
                self.memory.push(state, action, reward, next_state)

                state = next_state

                # Optimize model
                self.optimize_model()

                # Soft update target network's weights
                policy_net_dict = self.policy_net.state_dict()
                target_net_dict = self.target_net.state_dict()
                for key in policy_net_dict:
                    target_net_dict[key] = policy_net_dict[key] * self.tau + target_net_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_dict)

                # Done
                if done:
                    break;

    def test(self, n_episodes=100, n_steps=10, pretrained_model=None):
        """
        Test a pretrained model

        :param n_episodes: Total episodes to test for
        :param n_steps: Total steps or actions before each episode is terminated
        :param model_dir: Directory where model is saved

        :return: None
        """
        if pretrained_model:
            # Automatically load the latest saved model in the directory
            self.load_model(pretrained_model)

        n_success = 0
        n_failures = 0

        for episode in range(n_episodes):
            state, info = self.env.reset()

            for step in range(n_steps):
                state_img = extract_state_img(self.env, state, transforms=TRANSFORMS).to(DEVICE)
                action = torch.argmax(self.policy_net(state_img.unsqueeze(dim=0))).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                if (reward > 0):
                    n_success += 1
                elif (reward == 0 and terminated):
                    n_failures += 1

                done = terminated or truncated
                if terminated:
                    next_state = None

                title = f"Episode: {episode}, Step: {step}"
                # render_env(self.env, title)
                state = next_state

                if done:
                    break

        print(f"Accuracy: {n_success / n_episodes}")
        print(f"Failures: {n_failures / n_episodes}")
        print(f"Truncations: {(n_episodes - n_success - n_failures) / n_episodes}")

    def optimize_model(self):
        """
        Optimize the model based on loss between policy net and target net

        :params: None
        :return: None
        """
        # Sample random batch of states
        transitions = self.memory.sample()
        if transitions is None:
            return

        batch = Transition(*zip(*transitions))
        reward_batch = torch.tensor([reward for reward in batch.reward]).to(DEVICE)
        action_batch = torch.tensor([action for action in batch.action]).to(DEVICE)
        action_batch = torch.reshape(action_batch, (self.memory.batch_size, 1))

        # Get Q values predicted by the policy net
        current_state_imgs = torch.zeros(size=(len(batch.state), 1, IMG_WIDTH, IMG_WIDTH), dtype=torch.float).to(DEVICE)
        for i, state in enumerate(batch.state):
            current_state_imgs[i] = extract_state_img(self.env, state, transforms=TRANSFORMS)
        predicted_q_values = self.policy_net(current_state_imgs).gather(1, action_batch)

        # Get Q values as predicted by the target net
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE,
                                      dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None]).to(DEVICE)
        next_state_imgs = torch.zeros(size=(len(non_final_next_states), 1, IMG_WIDTH, IMG_WIDTH), dtype=torch.float).to(
            DEVICE)
        for i, state in enumerate(non_final_next_states):
            next_state_imgs[i] = extract_state_img(self.env, state, transforms=TRANSFORMS)

        expected_q_values = torch.zeros(size=(len(batch.state),)).to(DEVICE)
        with torch.no_grad():
            expected_q_values[non_final_mask] = self.target_net(next_state_imgs).max(1)[0]

        expected_q_values = (expected_q_values * self.gamma) + reward_batch
        expected_q_values = expected_q_values.unsqueeze(1)

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted_q_values, expected_q_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_model(self, checkpoint_num):
        """
        Save a model

        :param dir: Directory where the model is to be saved
        :param checkpoint_num: Include a checkpoint number in the model's name

        :return: None
        """
        # Set up path
        timestamp = int(datetime.now().timestamp())
        filename = "model_" + str(checkpoint_num) + "_" + str(timestamp) + ".pt"
        path = os.path.join('checkpoints', filename)

        # Save state dicts for policy and target net
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, filename):
        """
        Load a model

        :param dir: Directory where the model is saved

        :return: None
        """

        # Load model
        state_dicts = torch.load(filename, map_location=DEVICE)
        self.policy_net.load_state_dict(state_dicts['policy_net_state_dict'])
        self.target_net.load_state_dict(state_dicts['target_net_state_dict'])
        self.optimizer.load_state_dict(state_dicts['optim_state_dict'])

        # Put models in eval mode
        self.policy_net.eval()
        self.target_net.eval()


# DQN Networks
policy_net = DQN(IMG_WIDTH * IMG_WIDTH, ACTION_SPACE_SIZE).to(DEVICE)
target_net = DQN(IMG_WIDTH * IMG_WIDTH, ACTION_SPACE_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
target_net.load_state_dict(policy_net.state_dict())

# Initialize
env.reset()
dqn_trainer = Agent(env, policy_net, target_net, optimizer)

# Train
# dqn_trainer.train(20000, 10, resume_training=False)

# Test
dqn_trainer.test(n_episodes=100, n_steps=20, pretrained_model='model_best.pt')
