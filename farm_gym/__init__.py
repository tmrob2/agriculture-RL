from gym.envs.registration import register

register(
    id='Farming-v0',
    entry_point='farm_gym.envs:IrrigationEnv',
    max_episode_steps=300
)