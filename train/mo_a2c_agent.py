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
import argparse


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_logs = "/home/tmrob2/PycharmProjects/farming-gym/model_logs/"
tboardpth = "logs/"
train_writer = tf.summary.create_file_writer(tboardpth + f"/A2C-farm-gym-{dt.datetime.now().strftime('%d%m%Y%H%M')}")
beta = 100.
tboardpth = "logs/"
nsteps = 100000
env_id = 'Farming-v0'
NUM_AGENTS = 2
NUM_TASKS = 2
gamma = 1.0
e = [5., 7.]
mu = 0.5
c = 90. * 5  # two production cycles of full crops
chi = 0.
lam = 1.

def maize_quota(crops_planted, date):
    if date < datetime.datetime(2010, 1, 1).date():
        if crops_planted and 'maize' in crops_planted.keys():
            if crops_planted['maize']['qty_tagp'] > 6:
                return 1, 0.
            else:
                return 0, crops_planted['maize']['delta']
        else:
            return 0, 0.
    else:
        return 3, 0.

def wheat_quota(crops_planted, date):
    if date < datetime.datetime(2010, 1, 1).date():
        if crops_planted and 'wheat' in crops_planted.keys():
            if crops_planted['wheat']['qty_tagp'] > 10:
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
    end_date="2007-12-20",
    fixed_location=(-33.385300, 148.007904),  # Forbes NSW,
    beta=beta
)

farm2: gym.Env = gym.make(
    'Farming-v0',
    soil_type="EC2",
    start_date="2006-01-01",
    end_date="2007-12-20",
    fixed_location=(-36.626230, 142.188370),  # Wimmera Victoria
    beta=beta
)

train_writer = tf.summary.create_file_writer(tboardpth + f"/A2C-farm-gym-{dt.datetime.now().strftime('%d%m%Y%H%M')}")
envs = [farm1, farm2]
envs_ = mo_alg.EnvMem(envs, tasks)

# number of actions
n = envs[0].action_space.n
episode_reward = [0. for k in range(NUM_AGENTS + NUM_TASKS)]
state = [envs_.tf_reset(k) for k in range(NUM_AGENTS)]

def train_step0(episode, models):
    action_probs_l = []
    values_l = []
    rewards_l = []
    returns_l = []
    with tf.GradientTape() as tape:
        for agent in range(NUM_AGENTS):
            initial_state = envs_.tf_reset(agent)

            # Run the models for one episode to collect training data
            action_probs, values, rewards = envs_.run_episode(
                initial_state, models[agent], 1500, agent, 0
            )

            # Calculate the expected returns
            returns = envs_.get_expected_returns(rewards, gamma)

            action_probs_l.append(action_probs)
            values_l.append(values)
            rewards_l.append(rewards)
            returns_l.append(returns)

        ini_values = tf.convert_to_tensor([x[0, :] for x in values_l])
        loss_l = []
        for i in range(NUM_AGENTS):
            # convert the training data to appropriate TF tensor shapes
            action_probs = tf.expand_dims(action_probs_l[i], 1)
            values = values_l[i]
            returns = returns_l[i]

            # Calculating loss values to update out network
            ini_values_i = ini_values[i, :]
            loss = mo_alg.compute_loss(action_probs, values, returns, ini_values, ini_values_i, lam, chi, mu, e, c)
            with train_writer.as_default():
                tf.summary.scalar(f'agent_{i}_loss', loss, episode)
            loss_l.append(loss)
    # Compute the gradients from the loss vector
    vars_l = [m.trainable_variables for m in models]
    grads_l = tape.gradient(loss_l, vars_l)

    # Apply the gradients to the model params
    grads_l_f = [x for y in grads_l for x in y]
    vars_l_f = [x for y in vars_l for x in y]
    optimizer.apply_gradients(zip(grads_l_f, vars_l_f))

    episode_reward_l = [tf.math.reduce_sum(rewards_l[i], 0) for i in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        with train_writer.as_default():
            tf.summary.scalar(f'agent_{i} rewards', episode_reward_l[i][0], episode)
            tf.summary.scalar(f'agent_{i} task_0 rewards', episode_reward_l[i][1], episode)
            tf.summary.scalar(f'agent_{i} task_1 rewards', episode_reward_l[i][2], episode)

    return episode_reward_l[0][0], ini_values

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="resume training", action='store_true')
    args = parser.parse_args()

    min_episodes_criterion = 100
    max_episodes = 10000  # 10000
    max_steps_per_episode = 50  # 1000
    if args.resume:
        models = []
        for agent in range(NUM_AGENTS):
            model_path = os.path.join(model_logs, f"farming_agent{agent}/")
            models.append(tf.keras.models.load_model(model_path))
    else:
        models = [NNModel(n, NUM_TASKS) for _ in range(NUM_AGENTS)]

    running_reward = 0.
    threshold = 900.

    ## No discount
    # Discount factor for future rewards
    gamma = 1.00  # 0.99

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            # initial_state = tf.constant(env.reset(), dtype=tf.float32)
            # episode_reward = int(train_step(
            #    initial_state, models, optimizer, gamma, max_steps_per_episode))
            episode_reward, ini_values = train_step0(i, models)

            episode_reward = int(episode_reward)

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 20 == 0:
                for k in range(NUM_AGENTS):
                    print(f'values at the initial state for model#{k}: {ini_values[k]}')
                    # pass # print(f'Episode {i}: average reward: {avg_reward}')
                [m.save(os.path.join(model_logs, f"farming_agent{k}")) for k, m in enumerate(models)]

            if running_reward > threshold and i >= min_episodes_criterion:
                break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
    [m.save(os.path.join(model_logs, f"farming_agent{k}")) for k, m in enumerate(models)]
    print('Models saved')


