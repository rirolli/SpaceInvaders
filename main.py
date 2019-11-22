import gym              # palestra di allenamento
import tensorflow as tf # Deep Learning library
import numpy as np      # Handle matrices
import matplotlib.pyplot as plt
import random
import warnings

from skimage import transform # utile per processare i frames
from skimage.color import rgb2gray
from collections import deque
from DQNetwork import DQNetwork
from Memory import Memory

env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)

#Environment
obs_space = env.observation_space
frame_space_processed = [110, 84, 4]
action_space = env.action_space.n   # 6
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
stack_size = 4
episode_render = True

# Hyperparametri
gamma = 0.99     # 0 <= y <  1
alpha = 1e-4     # 0 <  a <= 1  learning rate

# Training hyperparameters
total_episodes = 100
max_steps = 5000
batch_size = 64

# Memory hyperparameters
pretrain_length = batch_size
memory_size = 1000000   # numero massimo di esperienze che la memoria può memorizzare

net = DQNetwork(frame_space_processed, action_space, alpha)
memory = Memory(max_size=memory_size)

print("possible_actions:\n", possible_actions)
print("Dimensioni Frame:\n", obs_space)                              # Box(210, 160, 3)
print("Spazio di azione:\n", action_space)                           # 6
print("Possibili azioni:\n", env.unwrapped.get_action_meanings())    # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalizza i valori dei pixel
    normalized_frame = cropped_frame / 255.0

    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame  # 110x84x1 frame

stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


# Instantiate memory
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()

    state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(choice)

    # env.render()

    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    # If the episode is finished (we're dead 3x)
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        state = env.reset()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our new state is now the next_state
        state = next_state

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i_episode in range(total_episodes):
        state = env.reset()

        episode_rewards = []

        state, stacked_frames = stack_frames(stacked_frames, state, True)
        print("==============================================")
        for t in range(max_steps):

            if episode_render:
                env.render()

            # Otteniamo Q
            q = sess.run(net.output, feed_dict={net.input: state.reshape((1, *state.shape))})
            # print("q:\n", str(q))
            choice = np.argmax(q)
            action = possible_actions[choice]
            # print("choice \n", str(choice))
            next_state, reward, done, _ = env.step(choice)
            # print(done, reward)
            episode_rewards.append(reward)

            if done:
                next_state = np.zeros((110, 84), dtype=np.int)
                # Ottieni il reward totale dell'episodio
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                total_reward = np.sum(episode_rewards)
                memory.add((state, action, reward, next_state, done))
                print('Total reward: {}'.format(total_reward),
                      "\nEpisodio finito dopo {} timesteps".format(t+1),
                      "\nTraining Loss {:.4f}".format(loss))
                break
            else:
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                memory.add((state, action, reward, next_state, done))
                state = next_state

            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            Qs_next_state = sess.run(net.output, feed_dict={net.input: next_states_mb})
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([net.loss, net.trainer],
                               feed_dict={net.input: states_mb,
                                          net.target_Q: targets_mb,
                                          net.actions: actions_mb})

env.close()