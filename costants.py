
# AMBIENTE
ENV_NAME = "SpaceInvaders-v0"
RENDER = True
NUM_EPISODES = 1000000
POST_PROCESS_IMAGE_SIZE = (105, 80, 1)
NUM_FRAMES = 4

# VALORI DI APPRENDIMENTO
MAX_EPSILON = 1
MIN_EPSILON = 0.1
EPSILON_MIN_ITER = 500000
GAMMA = 0.99
BATCH_SIZE = 32
TAU = 0.08
DELAY_TRAINING = 50000

# PATH DI SALVATAGGIO
STORE_PATH = "TensorBoard/DuelingQSI_{session}"
RECORD_PATH = "./recording/Record_{session}"
GIF_PATH = "./recording/gif_{session}/gif-{episode}.gif"
CKPT_PATH = "./checkpoints/{type}Network"
PARAMETERS_PATH = "./checkpoints/parameters.txt"

TRAIN = False

# SALVATAGGIO
LOAD = False
SAVE = True
SAVE_EACH = 1
