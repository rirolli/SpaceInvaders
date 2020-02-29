import gym
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime as dt

from costants import *
from model import DQModel
from memory import Memory
from wrappers import inizialize_wrapper
from saver import Saver


env = gym.make(ENV_NAME)
num_actions = env.action_space.n

online_network = DQModel(256, num_actions, str_name="Online")
target_network = DQModel(256, num_actions, str_name="Target")
online_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
# rendi target_network = primary_network
for t, e in zip(target_network.trainable_variables, online_network.trainable_variables):
    t.assign(e)

online_network.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())

saver = Saver(ckpt_path=CKPT_PATH, parameters_path=PARAMETERS_PATH)

if LOAD:
    episode, total_steps, eps, session = saver.load_parameters()
    saver.load_models(online_network, target_network)

    total_steps += 1
    episode += 1
else:
    episode = 0   # ogni volta che step % SAVE_EACH==0 vengono salvati i weights dei due modelli
    total_steps = 0
    eps = MAX_EPSILON
    session = dt.datetime.now().strftime('%d%m%Y%H%M')

memory = Memory(50000)
env = inizialize_wrapper(env=env, frame_skip=1, frame_height=POST_PROCESS_IMAGE_SIZE[0],
                         frame_width=POST_PROCESS_IMAGE_SIZE[1], record_path=RECORD_PATH.format(session=session))
delay_steps = 0
train_writer = tf.summary.create_file_writer(STORE_PATH.format(session=session))


def linear_eps_decay(steps: int):
    if steps < EPSILON_MIN_ITER:
        eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * (MAX_EPSILON - MIN_EPSILON)
    else:
        eps = MIN_EPSILON
    return eps


def choose_action(state, online_network, eps: float, delay_steps: int):
    if delay_steps < DELAY_TRAINING:
        return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(online_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0],
                                                               POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy()))


def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack


print(f"\n -- Creazione file di Tensorboard-log: \"{STORE_PATH}/DuelingQSI_{dt.datetime.now().strftime('%d%m%YH%M')}\" -- \n")

for i in range(NUM_EPISODES):
    state = env.reset()
    state_stack = tf.Variable(np.repeat(state, NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                    POST_PROCESS_IMAGE_SIZE[1],
                                                                    NUM_FRAMES)))
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    images = [] #
    while True:
        if RENDER:
            env.render()
        action = choose_action(state=state_stack, online_network=online_network, eps=eps, delay_steps=delay_steps)
        next_state, reward, done, info = env.step(action=action)
        tot_reward += reward
        state_stack = process_state_stack(state_stack=state_stack, state=next_state)
        # salva in memory il nuovo stato
        memory.add_sample(frame=next_state, action=action, reward=reward, done=done)

        if delay_steps > DELAY_TRAINING:
            loss = online_network.train_model(memory=memory, target_network=target_network)
            online_network.update_network(target_network=target_network)
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=total_steps)
        else:
            loss = -1
        avg_loss += loss

        # decresce il valore di eps in modo lineare
        if delay_steps > DELAY_TRAINING:
            eps = linear_eps_decay(steps=total_steps)

        delay_steps += 1
        total_steps += 1

        if done:
            if delay_steps > DELAY_TRAINING:
                avg_loss /= cnt
                print(f"Episodio: {episode}, Reward: {tot_reward}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', tot_reward, step=episode)
                    tf.summary.scalar('avg loss', avg_loss, step=episode)
                    tf.summary.scalar('eps', eps, step=episode)

                if episode % SAVE_EACH == 0 and SAVE:
                    saver.save_models(episode, online_network, target_network)
                    saver.save_parameters(total_steps=total_steps, episode=episode, eps=eps, session=session)
                episode += 1
            else:
                print(f"Pre-training...Episodio: {i}")
            break

        cnt += 1

# Quando finisce il numero degli episodi allora salva un checkpoint
if SAVE:
    saver.save_models(episode, online_network, target_network)
    saver.save_parameters(total_steps=total_steps, episode=episode, eps=eps)
