import gym              # palestra di allenamento
import tensorflow as tf # libreria di Deep Learning
import numpy as np      # Manipolazione matrici
import time

from collections import deque

from FrameHelper import FrameHelper
from agent import DQNAgent
from wrappers import inizialize_wrapper

from variables import *     # Contiene tutte le costanti e gli hyperparametri

# INIZIALIZZAZIONE AMBIENTE DI GIOCO
env = gym.make(env_name)
env = inizialize_wrapper(env)

# parametri d'ambiente
action_space = env.action_space.n   # 6
obs_space = env.observation_space   # Box(210, 160, 3) non processato!

# INIZIALIZZAZIONI CLASSI
frameH = FrameHelper(stack_size=stack_size)
agent = DQNAgent(state_size=frame_space_processed, action_space=action_space, max_memory=memory_size,
                 learning_rate=alpha, gamma=gamma, double_q=True)


stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4) # creazione di uno stack di frame VUOTO
tf.reset_default_graph()    # resetta il grafico
rewards = []                # una lista di reward
step = 0                    # inizializza gli step
start = time.time()         # inizializza il tempo di inizio
episode_frames = []         # memorizza tutti i frames dell'episodio per creare una GIF

for episode in range(total_episodes):

    state = env.reset()
    state, stacked_frames = frameH.stack_frames(stacked_frames, state, True)

    total_reward = 0
    iter = 0

    while True:

        if episode_render and not cluster:
            env.render()

        action = agent.run(state=state)

        next_state, reward, done, _ = env.step(action=action)

        next_state, stacked_frames = frameH.stack_frames(stacked_frames, next_state, False)

        agent.add(experience=(state, next_state, action, reward, done))

        agent.learn()

        total_reward += reward

        state = next_state

        iter += 1

        if done:
            break

    rewards.append(total_reward/iter)

    if episode % 25 == 0:
        print('Episode {e} - '
              'Frame {f} - '
              'Frames/sec {fs} - '
              'Epsilon {eps} - '
              'Mean Reward {r}'.format(e=episode,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        start = time.time()
        step = agent.step

    np.save('rewards.npy', rewards)

env.close()



