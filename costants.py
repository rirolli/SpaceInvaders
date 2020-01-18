import datetime as dt
import os

ENV_NAME = "SpaceInvaders-v0"
RENDER = True
STORE_PATH = 'TensorBoard'
RECORD_PATH = f"./recording/Record_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
NUM_EPISODES = 1000000
MAX_EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_MIN_ITER = 500000
GAMMA = 0.99
BATCH_SIZE = 32
TAU = 0.08
POST_PROCESS_IMAGE_SIZE = (105, 80, 1)
DELAY_TRAINING = 50000
NUM_FRAMES = 4
DOUBLE_Q = True
