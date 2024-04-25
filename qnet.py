# Deep Q Learning / Keras / OpenAI / Frozen Lake / Not Slippery / 5x5
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

custom_map = [
    'SFFHF',
    'HFHFF',
    'HFFFH',
    'HHHFH',
    'HFFFG'
]

map_small = [
    'SFFF',
    'FHFF',
    'FFFF',
    'FFFG',
]

env = gym.make("FrozenLake-v1", desc=map_small, is_slippery=False)
train_episodes = 30
test_episodes = 100
max_steps = 50
state_size = env.observation_space.n
action_size = env.action_space.n
batch_size = 10  # 32


class Agent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2500)
        self.learning_rate = 0.001
        self.epsilon = 1
        self.max_eps = 1
        self.min_eps = 0.01
        # self.eps_decay = 0.001/3
        # self.eps_decay = 0.005
        self.eps_decay = 1 / train_episodes
        self.gamma = 0.9
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon_lst = []
        self.model = self.buildmodel()

    def buildmodel(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, new_state, reward, term, trunc, state, action):
        self.memory.append((new_state, reward, term, trunc, state, action))

    def action(self, state):
        # if np.random.rand() < self.epsilon:
        #     return np.random.randint(0, 4)
        # return np.argmax(self.model.predict(state))
        return env.action_space.sample()

    def pred(self, state):
        print(self.model.predict(state))
        return np.argmax(self.model.predict(state))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for new_state, reward, term, trunc, state, action in minibatch:
            target = reward
            if not term or not trunc:
                target = reward + self.gamma * np.amax(self.model.predict(new_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon = (self.max_eps - self.min_eps) * np.exp(-self.eps_decay * episode) + self.min_eps

        self.epsilon_lst.append(self.epsilon)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = Agent(state_size, action_size)

load = False
if load:
    agent.load("testing123")

reward_lst = []
for episode in range(train_episodes):
    if load:
        break
    start = time.time()
    state = env.reset()
    state = state[0]
    state_arr = np.zeros(state_size)
    state_arr[state] = 1
    state = np.reshape(state_arr, [1, state_size])
    reward = 0
    for t in range(max_steps):
        # env.render()
        action = agent.action(state)
        new_state, reward, term, trunc, info = env.step(action)
        new_state_arr = np.zeros(state_size)
        new_state_arr[new_state] = 1
        new_state = np.reshape(new_state_arr, [1, state_size])
        agent.add_memory(new_state, reward, term, trunc, state, action)
        state = new_state

        if term or trunc:
            end = time.time()
            print(
                f'Episode: {episode:4}/{train_episodes} and step: {t:4}. Eps: {float(agent.epsilon):.2}, reward {reward}, {end - start} secs')
            break

    reward_lst.append(reward)

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

print(' Train mean % score= ', round(100 * np.mean(reward_lst), 1))

agent.model.save("testing123")

# test
env = gym.make("FrozenLake-v1", desc=map_small, is_slippery=False, render_mode="human")
test_wins = []
for episode in range(test_episodes):
    state = env.reset()
    state = state[0]
    state_arr = np.zeros(state_size)
    state_arr[state] = 1
    state = np.reshape(state_arr, [1, state_size])
    print('******* EPISODE ', episode, ' *******')

    for step in range(max_steps):
        # env.render()
        action = agent.pred(state)
        new_state, reward, term, trunc, info = env.step(action)
        new_state_arr = np.zeros(state_size)
        new_state_arr[new_state] = 1
        new_state = np.reshape(new_state_arr, [1, state_size])
        state = new_state
        if term or trunc:
            if reward == 1:
                print(f"GOAL")
                test_wins.append(reward)
            break

    test_wins.append(0)
env.close()

print(' Test mean % score= ', int(100 * np.mean(test_wins)))

fig = plt.figure(figsize=(10, 12))
matplotlib.rcParams.clear()
matplotlib.rcParams.update({'font.size': 16})
plt.subplot(311)
plt.scatter(list(range(len(reward_lst))), reward_lst, s=0.2)
plt.title('5x5 Frozen Lake Result(DQN) \n \nTrain Score')
plt.ylabel('Score')
plt.xlabel('Episode')

plt.subplot(312)
plt.scatter(list(range(len(agent.epsilon_lst))), agent.epsilon_lst, s=0.2)
plt.title('Epsilon')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

plt.subplot(313)
plt.scatter(list(range(len(test_wins))), test_wins, s=0.5)
plt.title('Test Score')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.ylim((0, 1.1))
plt.savefig('5x5resultdqn.png', dpi=300)
plt.show()
