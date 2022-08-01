import argparse
import gym
import farm_gym
from mo_rl import algorithm as alg
from mo_rl.model import NNModel
import datetime as dt
import tensorflow as tf
import numpy as np
from tensorflow import keras

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
beta = 10.
tboardpth = "logs/"
nsteps = 100000
env_id = 'Farming-v0'
env_kwargs = dict(
    soil_type="EC4",
    fixed_date="2006-01-01",
    fixed_location=(-33.385300, 148.007904),
    intervention_interval=7, # is the farming period between interventions
    beta=beta
)
train_writer = tf.summary.create_file_writer(tboardpth + f"/A2C-farm-gym-{dt.datetime.now().strftime('%d%m%Y%H%M')}")
env = gym.make(env_id, **env_kwargs)
env_ = alg.EnvMem(env)

def train():
    episode_reward_sum = 0.0

    # construct the environment either from memory or initialise
    # setup environment
    # loss = tf.constant(0.)
    model = NNModel(env.action_space.n)
    state = tf.constant(env.reset(), dtype=tf.float32)
    initial_state_shape = state.shape
    episode = 1
    with tf.GradientTape(persistent=True) as tape:
        for step in range(nsteps):
            rewards = tf.TensorArray(dtype=tf.float32, size=alg.BATCH_SIZE)
            values = tf.TensorArray(dtype=tf.float32, size=alg.BATCH_SIZE)
            dones = tf.TensorArray(dtype=tf.int32, size=alg.BATCH_SIZE)
            action_probs = tf.TensorArray(dtype=tf.float32, size=alg.BATCH_SIZE)
            for k in tf.range(alg.BATCH_SIZE):
                state = tf.expand_dims(state, 0)
                policy_logits, value = model.call(state)
                action = tf.random.categorical(policy_logits, 1)[0, 0]
                action_probs_t = tf.nn.softmax(policy_logits, 1)
                action_probs = action_probs.write(k, action_probs_t[0, action])
                state, reward, done = env_.tf_env_step(action)
                values = values.write(k, value)
                dones = dones.write(k, done)
                episode_reward_sum += reward
                state.set_shape(initial_state_shape)
                if tf.cast(done, tf.bool):
                    rewards = rewards.write(k, 0.0)
                    state = env_.tf_reset()
                    state.set_shape(initial_state_shape)
                    print(f"Episode: {episode},latest episode reward: {episode_reward_sum}, loss: {loss}")
                    with train_writer.as_default():
                        tf.summary.scalar('rewards', episode_reward_sum, episode)
                    episode_reward_sum = 0.
                    episode += 1
                else:
                    rewards = rewards.write(k, reward)
            state = tf.expand_dims(state, 0)
            _, next_value = model.call(state)
            next_value = tf.squeeze(next_value)
            state = tf.squeeze(state)
            rewards = rewards.stack()
            values = values.stack()
            dones = dones.stack()
            action_probs = action_probs.stack()
            values = tf.squeeze(values)
            returns, advs = alg.advantages(rewards, dones, values, next_value)
            loss = alg.compute_loss(action_probs, advs, values, returns)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            with train_writer.as_default():
                tf.summary.scalar('tot_loss', loss, step)
    del tape

train()

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--name", help="model name", default="", type=str, required=False)
#    parser.add_argument("--beta", type=float, default=10., help="penalty for fertilization")
#    parser.add_argument("--tensorboard", help="Tensorboard log dir", default="", type=str)
#    parser.add_argument("--log", help="directory to save model", default="", type=str)
#    parser.add_argument("--n_steps", type=int, default=5e5, help="number of timesteps to train for")
#    parser.add_argument("--resume", help="resume training", action='store_true')
#    args = parser.parse_args()
#
#    train(args.name, args.log, args.tensorboard, args.beta, args.n_steps, args.resume)