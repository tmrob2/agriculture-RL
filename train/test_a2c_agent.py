import argparse
import gym
import farm_gym
from mo_rl import algorithm as alg
from mo_rl.model import NNModel
import datetime as dt
import tensorflow as tf
import numpy as np
from tensorflow import keras

def train(name, log, tboardpth, beta, nsteps, resume):
    epsisode_reward_sum = 0

    # construct the environment either from memory or initialise
    # setup environment
    env_kwargs = dict(
        soil_type="EC4", 
        fixed_date="2006-01-01", 
        fixed_location=(-33.385300, 148.007904),
        intervention_interval=7, # is the farming period between interventions
        beta=beta
    )

    env_id = 'Farming-v0'
    env = gym.make(env_id, **env_kwargs)

    model = NNModel(env.action_space.n)
    model.compile(optimizer=keras.optimizers.Adam(), loss=[alg.critic_loss, alg.actor_loss])
    train_writer = tf.summary.create_file_writer(tboardpth + f"/A2C-farm-gym-{dt.datetime.now().strftime('%d%m%Y%H%M')}")
    state = env.reset()
    episode = 1

    for step in range(nsteps):
        rewards = []
        actions = []
        values = []
        states = []
        dones = []
        for _ in range(alg.BATCH_SIZE):
            _, policy_logits = model(state.reshape(1, -1))
            action, value = model.action_value(state.reshape(1, -1))
            new_state, reward, done, _info = env.step(action.numpy()[0])
            actions.append(action)
            values.append(value.numpy()[0])
            states.append(state)
            dones.append(done)
            epsisode_reward_sum += reward
            state = new_state
            if done:
                rewards.append(0.0)
                state = env.reset()
                print(f"Episode: {episode},latest episode reward: {epsisode_reward_sum}, loss: {loss}")
                with train_writer.as_default():
                    tf.summary.scalar('rewards', epsisode_reward_sum, episode)
                    epsisode_reward_sum = 0
                    episode += 1
            else:
                rewards.append(reward)
        _, next_value = model.action_value(state.reshape(1, -1))
        discounted_rewards, advs = alg.advantages(rewards, dones, values, next_value.numpy()[0])

        # combine the actions and advantages into a comvbined array for passing
        # actor_loss function
        combined = np.zeros((len(actions), 2))
        combined[:, 0] = actions
        combined[:, 1] = advs

        loss = model.train_on_batch(tf.stack(states), [discounted_rewards, combined])

        with train_writer.as_default():
            tf.summary.scalar('tot_loss', np.sum(loss), step)


train("test", "model_logs/", "logs/", 10., 100000, False)

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