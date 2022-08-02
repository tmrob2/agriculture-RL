import datetime

import tensorflow as tf
import gym
from mo_rl import mo_alg
from mo_rl.dfa import DFA
from mo_rl.model import NNModel
import datetime as dt
import collections
import statistics
import farm_gym
import tqdm
import os


optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
model_logs = "model_logs/"
tboardpth = "logs/"
beta = 10.
tboardpth = "logs/"
nsteps = 100000
env_id = 'Farming-v0'
NUM_AGENTS = 2
NUM_TASKS = 2
gamma = 1.0
e = [12, 18]
mu = 0.5
c = 10.
chi = 1.
lam = .1

def maize_quota(crops_planted, date):
    if date < datetime.datetime(2008, 1, 1).date():
        if crops_planted and 'maize' in crops_planted.keys():
            if crops_planted['maize']['qty_tagp'] > 12:
                return 1, 0.
            else:
                return 0, crops_planted['maize']['delta']
        else:
            return 0, 0.
    else:
        return 3, 0.

def wheat_quota(crops_planted, date):
    if date < datetime.datetime(2008, 1, 1).date():
        if crops_planted and 'wheat' in crops_planted.keys():
            if crops_planted['wheat']['qty_tagp'] > 18:
                return 1, 0.
            else:
                return 0, crops_planted['wheat']['delta']
        else:
            return 0, 0.
    else:
        return 3, 0.

def done(*args):
    return 2, 0.

def finish(*args):
    return 2, 0.

def fail(*args):
    return 3, 0.

def maize_contract():
    dfa = DFA(0, [2], [3])
    dfa.add_state(0, maize_quota)
    dfa.add_state(1, done)
    dfa.add_state(2, finish)
    dfa.add_state(3, fail)
    return dfa

def wheat_contract():
    dfa = DFA(0, [2], [3])
    dfa.add_state(0, wheat_quota)
    dfa.add_state(1, done)
    dfa.add_state(2, finish)
    dfa.add_state(3, fail)
    return dfa

maize_ = maize_contract()
wheat_ = wheat_contract()
tasks = [maize_, wheat_]

farm1: gym.Env = gym.make(
    'Farming-v0',
    soil_type="EC4",
    start_date="2006-01-01",
    end_date="2009-12-20",
    fixed_location=(-33.385300, 148.007904)  # Forbes NSW
)

farm2: gym.Env = gym.make(
    'Farming-v0',
    soil_type="EC2",
    start_date="2006-01-01",
    end_date="2009-12-20",
    fixed_location=(-36.626230, 142.188370)  # Wimmera Victoria
)

envs = [farm1, farm2]
envs_ = mo_alg.EnvMem(envs, tasks)

# number of actions
n = envs[0].action_space.n
episode_reward = [0. for k in range(NUM_AGENTS + NUM_TASKS)]
models = []
for agent in range(NUM_AGENTS):
    model_path = os.path.join(model_logs, f"farming_agent{agent}/")
    models.append(tf.keras.models.load_model(model_path))

state = [envs_.tf_reset(k) for k in range(NUM_AGENTS)]

if __name__ == '__main__':
    for agent in range(NUM_AGENTS):
        print("Agent ", agent)
        initial_state = state[agent]

        _, values, rewards = envs_.run_episode(initial_state, models[agent], 1500, agent, 1)
        returns = envs_.get_expected_returns(rewards, gamma)
        print(envs_.data[agent][-1])
        print(returns[0])

