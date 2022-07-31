import gym
import farm_gym
from stable_baselines3.common.env_checker import check_env

env = gym.make(
    'Farming-v0', 
    soil_type="EC4", 
    fixed_date="2006-01-01", 
    fixed_location=(-33.385300, 148.007904)
)
print("Checking env...")
check_env(env, warn=True)
print(r"OK")