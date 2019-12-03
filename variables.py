
# AMBIENTE DI GIOCO
env_name = 'SpaceInvaders-v0'
episode_render = True
frame_space_processed = (84, 84, 4)
stack_size = 4
total_episodes = 1000000

# HYPERPARAMETRI DI APPRENDIMENTO
gamma = 0.90        # 0 <= y <  1
alpha = 0.00025     # 0 <  a <= 1  learnig rate

# HYPERPARAMETRI DI MEMORIA
memory_size = 1000000   # numero massimo di esperienze che la memoria può memorizzare

# RIPRISTINARE SALVATAGGIO DI APPRENDIMENTO
load_model = True

# ABILITA IMPOSTAZIONI CLUSTER
cluster = False

# PATH
model_path = './models/model'
load_models_path = './models'
tensorboard_path = "./tensorboard"

