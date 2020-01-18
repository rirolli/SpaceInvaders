import gym
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import datetime as dt

from costants import *
from model import DQModel
from memory import Memory
from wrappers import inizialize_wrapper


env = gym.make(ENV_NAME)
env = inizialize_wrapper(env=env, frame_skip=1, frame_height=POST_PROCESS_IMAGE_SIZE[0],
                         frame_width=POST_PROCESS_IMAGE_SIZE[1], record_path=RECORD_PATH)
num_actions = env.action_space.n

online_network = DQModel(256, num_actions)
target_network = DQModel(256, num_actions)
online_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
# rendi target_network = primary_network
for t, e in zip(target_network.trainable_variables, online_network.trainable_variables):
    t.assign(e)

online_network.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())

memory = Memory(50000)


def linear_eps_decay(steps: int):
    if steps < EPSILON_MIN_ITER:
        eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * (MAX_EPSILON - MIN_EPSILON)
    else:
        eps = MIN_EPSILON
    return eps


def choose_action(state, primary_network, eps, step):
    if step < DELAY_TRAINING:
        return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(primary_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0],
                                                                POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy()))


def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack


eps = MAX_EPSILON
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQSI_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
steps = 0

print(f"\n -- Creazione file di Tensorboard-log: \"{STORE_PATH}/DuelingQSI_{dt.datetime.now().strftime('%d%m%YH%M')}\" -- \n")
print(f" -- Allenamento iniziato in data: {dt.datetime.now().strftime('%d%m%Y%H%M')} -- ")

for i in range(NUM_EPISODES):
    state = env.reset()
    state_stack = tf.Variable(np.repeat(state, NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                    POST_PROCESS_IMAGE_SIZE[1],
                                                                    NUM_FRAMES)))
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    while True:
        if RENDER:
            env.render()
        action = choose_action(state_stack, online_network, eps, steps)
        next_state, reward, done, info = env.step(action)
        tot_reward += reward
        state_stack = process_state_stack(state_stack, next_state)
        # salva in memory il nuovo stato
        memory.add_sample(frame=next_state, action=action, reward=reward, done=done)

        if steps > DELAY_TRAINING:
            loss = online_network.train_model(memory, target_network)
            online_network.update_network(target_network)
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=steps)
        else:
            loss = -1
        avg_loss += loss

        # decresce il valore di eps in modo lineare
        if steps > DELAY_TRAINING:
            eps = linear_eps_decay(steps=steps)

        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                print(f"Episodio: {i}, Reward: {tot_reward}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
                    tf.summary.scalar('eps', eps, step=i)
            else:
                print(f"Pre-training...Episodio: {i}")
            break

        cnt += 1
