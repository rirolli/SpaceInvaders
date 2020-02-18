import datetime as dt

ENV_NAME = "SpaceInvaders-v0"
RENDER = True
STORE_PATH = "TensorBoard/DuelingQSI_{session}"
RECORD_PATH = "./recording/Record_{session}"
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
CKPT_PATH = "checkpoints/{type}Network/cp-{epoch:04d}.ckpt"
PARAMETERS_PATH = "checkpoints/parameters.txt"
LOAD = True
SAVE = True
SAVE_EACH = 1

