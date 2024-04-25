import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # Change is_slippery to True for the stochastic version

# Make the environment deterministic (optional)
# env.seed(0)
# np.random.seed(0)

# Wrap it for vectorized operations
vec_env = make_vec_env(lambda: env, n_envs=1)

# Define the DQN model
model = DQN('MlpPolicy', vec_env, verbose=1, buffer_size=10000, learning_rate=1e-3, batch_size=64,
            exploration_fraction=0.1, exploration_final_eps=0.02, target_update_interval=250)

# Train the model
model.learn(total_timesteps=100000)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the model
obs = env.reset()
obs = obs[0]
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, term, trunc, info = env.step(action)
    env.render()
    if term or trunc:
        obs = env.reset()

# Close the environment
env.close()
