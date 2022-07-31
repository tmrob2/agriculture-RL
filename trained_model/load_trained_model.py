import gym
import farm_gym
import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from pprint import pprint

def test(name, log_dir, beta):
    # setup environment
    env_kwargs = dict(
        soil_type="EC4", 
        fixed_date="2006-01-01", 
        fixed_location=(-33.385300, 148.007904),
        intervention_interval=7,
        beta=beta
    )
    tag = f"{name}_beta_{str(beta).replace('.', '_')}"
    model_path = os.path.join(log_dir, tag)
    stats_path = os.path.join(log_dir, f"{tag}.pkl")

    #env_kwargs = {'intervention_interval':7, 'beta':beta}
    env_id = 'Farming-v0'
    env = gym.make(env_id, **env_kwargs)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path)
    env = VecNormalize.load(stats_path, env)
    env.reset()
    model.set_env(env)

    obs = env.reset()
    for _ in range(0, 50):
        action, _state = model.predict(obs, deterministic=True)
        print(action)
        obs, _, done, info = env.step(action)
        if done:
            obs = env.reset()
            pprint(info)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="name", default="", type=str, required=False)
    parser.add_argument("--model", help="/path/to/model", default="model_logs", type=str, required=False)
    parser.add_argument("--beta", type=float, default=10., help="penalty for fertilization")
    args = parser.parse_args()
    test(args.name, args.model, args.beta)