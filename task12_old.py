import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
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

small_map = [
    'SFFF',
    'FHFF',
    'FFFF',
    'FFFG',
]

env = gym.make("FrozenLake-v1", desc=small_map, is_slippery=False)
train_episodes = 10  # 4000
test_episodes = 10
max_steps = 20  # 300
state_size = env.observation_space.n
action_size = env.action_space.n
batch_size = 32


class Agent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2500)
        self.learning_rate = 0.001
        self.epsilon = 1
        self.max_eps = 1
        self.min_eps = 0.01
        self.eps_decay = 0.001 / 3
        self.gamma = 0.98  # 0.9
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon_lst = []
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, new_state, reward, done, state, action):
        self.memory.append((new_state, reward, done, state, action))

    def action(self, state):
        if np.random.rand() > self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.model.predict(state))

    def predict(self, state):
        return np.argmax(self.model.predict(state))

    def replay(self, batch_size, episode):
        minibatch = random.sample(self.memory, batch_size)
        for new_state, reward, done, state, action in minibatch:
            target = reward
            if not done:
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


def run_simulation(training=True, episodes=1):
    # if not training:
    #     env = gym.make("FrozenLake-v1", desc=small_map, is_slippery=False, render_mode="human")
    # else:
    #     env = gym.make("FrozenLake-v1", desc=small_map, is_slippery=False)

    wins = []
    for episode in range(episodes):
        state = env.reset()
        state = state[0]
        state_arr = np.zeros(state_size)
        state_arr[state] = 1
        state = np.reshape(state_arr, [1, state_size])
        reward = 0

        states = []
        if not training:
            states.append(state)
            print('******* EPISODE ', episode, ' *******')

        for t in range(max_steps):
            # env.render()
            if training:
                action = agent.action(state)
            else:
                action = agent.predict(state)
            new_state, reward, term, trunc, info = env.step(action)
            new_state_arr = np.zeros(state_size)
            new_state_arr[new_state] = 1
            new_state = np.reshape(new_state_arr, [1, state_size])

            if training:
                agent.add_memory(new_state, reward, term, state, action)
            else:
                states.append(state)
            state = new_state

            if term:
                if training:
                    print(
                        f'Episode: {episode:4}/{episodes} and step: {t:4}. Eps: {float(agent.epsilon):.2}, reward {reward}')
                else:
                    print(reward)
                break

        wins.append(reward)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size, episode)

    print(' Train mean % score= ', round(100 * np.mean(wins), 1))
    return wins


def run_metrics(training_wins, testing_wins):
    fig = plt.figure(figsize=(10, 12))
    matplotlib.rcParams.clear()
    matplotlib.rcParams.update({'font.size': 16})
    plt.subplot(311)
    plt.scatter(list(range(len(training_wins))), training_wins, s=0.2)
    plt.title('Frozen Lake \n \nTrain Score')
    plt.ylabel('Score')
    plt.xlabel('Episode')

    plt.subplot(312)
    plt.scatter(list(range(len(agent.epsilon_lst))), agent.epsilon_lst, s=0.2)
    plt.title('Epsilon')
    plt.ylabel('Epsilon')
    plt.xlabel('Episode')

    plt.subplot(313)
    plt.scatter(list(range(len(testing_wins))), testing_wins, s=0.5)
    plt.title('Test Score')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.ylim((0, 1.1))
    plt.savefig('5x5resultdqn.png', dpi=300)
    plt.show()


training_wins = run_simulation(training=True, episodes=train_episodes)
testing_wins = run_simulation(training=False, episodes=test_episodes)
run_metrics(training_wins, testing_wins)

env.close()
