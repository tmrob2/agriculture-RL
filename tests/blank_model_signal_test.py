import gym
import numpy as np

import farm_gym

env: gym.Env = gym.make(
    'Farming-v0',
    soil_type="EC4",
    start_date="2006-01-01",
    end_date="2009-12-20",
    fixed_location=(-33.385300, 148.007904)
)


env.reset()

# We want to test the application of some actions

total_rewards = []
for t in range(1000): # this should generate 10 weeks of data or 10 intervention periods
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_rewards.append(reward)
    if done:
        break
print("total rewards ", np.sum(np.array(total_rewards)))