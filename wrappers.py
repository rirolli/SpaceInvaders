from gym.wrappers.monitor import Monitor
from gym.wrappers.atari_preprocessing import AtariPreprocessing

def inizialize_wrapper(env):
    """Applica un set di wrappers per i giochi Atari"""
    env = Monitor(env=env, directory="./recording", force=True)     # Registra delle partite in .mp4
    env = AtariPreprocessing(env=env)                               # preprocessa le immagini in (84, 84)
    return env
