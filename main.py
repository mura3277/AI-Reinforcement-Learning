import numpy as np
import gym
import random
import time
import matplotlib.pyplot as plt

from gymnasium.envs.toy_text.frozen_lake import generate_random_map

map = generate_random_map(size=6, p=0.3)
# map = ['SFHH', 'HFFH', 'HHFF', 'FHFG']
print(map)
# map = ['SFFHHHFFHH', 'FHFFHFHHHH', 'FHHFFFFFFF', 'HFHHFFHHFF', 'FHHFHHHHHF', 'HHHHHFHFFF', 'FHHHHHFFFF', 'HFFHHFFFHF', 'HHFHHHHFFF', 'HHFHHHHHFG']
env = gym.make("FrozenLake-v1", desc=map, is_slippery=False)

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

total_episodes = 50_000       # Total episodes
learning_rate = 0.4           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.99                # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.0005          # Exponential decay rate for exploration prob

# List of rewards
rewards = []

start = time.time()

rewards_per_episode = np.zeros(total_episodes)

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    state = state[0]
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        # First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info, _ = env.step(action)

        # we have fallen into a hole, apply penalty to the AI for going to these areas
        # if done and reward == 0:
        #     reward -= 1

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        MAX = np.max(qtable[new_state, :])
        A = (reward + gamma * MAX - qtable[state, action])
        qtable[state, action] = qtable[state, action] + learning_rate * A

        total_rewards += reward

        if reward == 1:
            rewards_per_episode[episode] = 1

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

sum_rewards = np.zeros(total_episodes)
for t in range(total_episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig(f"plt-ST({state_size})-EP({total_episodes})-L({learning_rate})-G({gamma}).png")

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)

end = time.time()
print(f"Elapsed: {end - start}s")
