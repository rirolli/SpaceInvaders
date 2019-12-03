from gym.wrappers.monitor import Monitor
from gym.wrappers.atari_preprocessing import AtariPreprocessing

from variables import cluster

def inizialize_wrapper(env):
    """ Applica un set di wrappers per i giochi Atari"""
    env = Monitor(env=env, directory="./recording", force=True)  # Registra delle partite in .mp4
    if cluster:
        env = AtariPreprocessing(frame_skip=1, env=env)  # preprocessa le immagini in (84, 84)
    else:
        env = AtariPreprocessing(env=env)  # preprocessa le immagini in (84, 84)
    return env
