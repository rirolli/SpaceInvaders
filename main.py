import gym              # palestra di allenamento
import tensorflow as tf # Deep Learning library
import numpy as np      # Handle matrices
import matplotlib.pyplot as plt
import random
import warnings

from skimage import transform # utile per processare i frames
from skimage.color import rgb2gray
from collections import deque

'''
warnings.filterwarnings('ignore')

# NOME AMBIENTE
env_name = 'SpaceInvaders-v0'

### MODEL HYPERPARAMETERS
state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n # 8 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


# Creiamo l'ambiente
env = gym.make(env_name)

print("Dimensioni Frame: ", env.observation_space)                  # Box(210, 160, 3)
print("Spazio di azione: ", env.action_space.n)                     # 6
print("Possibili azioni: ", env.unwrapped.get_action_meanings())    # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame  # 110x84x1 frame


# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


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
'''

env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)


#Environment
obs_space = env.observation_space
action_space = env.action_space.n
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
print("possible_actions:\n",possible_actions)
stack_size = 4

# Hyperparametri
gamma = 0.99    # 0 <= gamma <  1
alpha = 0.1     # 0 <  alpha <= 1


print("Dimensioni Frame: ", obs_space)                  # Box(210, 160, 3)
print("Spazio di azione: ", action_space)                     # 6
print("Possibili azioni: ", env.unwrapped.get_action_meanings())    # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']


def preprocess_frame(frame):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize Pixel Values
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

class Network:
    def __init__(self, state_size, action_size, learning_rate):
        # Input
        self.input = tf.placeholder(tf.float32, [None, *state_size],name="input")
        self.conv1 = tf.layers.conv2d(inputs=self.input,
                                      filters=32,
                                      kernel_size=[8, 8],
                                      strides=[4,4],
                                      padding="VALID",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv1")

        self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

        self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                      filters=64,
                                      kernel_size=[4, 4],
                                      strides=[2, 2],
                                      padding="VALID",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv2")

        self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

        self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                      filters=64,
                                      kernel_size=[3, 3],
                                      strides=[2, 2],
                                      padding="VALID",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv3")

        self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

        self.flatten = tf.contrib.layers.flatten(self.conv3_out)

        self.fc = tf.layers.dense(inputs=self.flatten,
                                  units=512,
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")

        self.output = tf.layers.dense(inputs=self.fc,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      units=action_size,
                                      activation=None)

        self.action_chosen = tf.argmax(self.output, 1)
        #---

        self.actions = tf.placeholder(tf.float32, [None, action_size], name="actions_")

        # Q is our predicted Q value.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))


net = Network([110, 84, 4], action_space, alpha)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i_episode in range(100):
        state = env.reset()
        episode_rewards = []
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        for t in range(5000):
            env.render()
            qs = sess.run(net.output, feed_dict={net.input: state.reshape((1, *state.shape))})
            print("qs:\n",str(qs))
            choice = np.argmax(qs)
            #action = possible_actions[choice]
            print("choice \n",str(choice))
            #print("action\n",str(action))
            next_state, reward, done, _ = env.step(choice)
            print(done, reward)
            episode_rewards.append(reward)
            if done:
                next_state = np.zeros((110, 84), dtype=np.int)

                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)

                print('Total reward: {}'.format(total_reward))

                print("Episode finished after {} timesteps".format(t+1))
                break
            else:
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state

env.close()