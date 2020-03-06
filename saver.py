import os
import shutil

import tensorflow as tf
import datetime as dt

from model import DQModel
from json_helper import JsonHelper
from costants import MAX_EPSILON


class Saver:
    def __init__(self, ckpt_path: str, parameters_path: str):
        self.ckpt_path = ckpt_path
        self.parameters_path = parameters_path
        self.parameters_helper = JsonHelper(self.parameters_path)

    def save_models(self, episode: int, *models: DQModel):
        for model in models:
            try:
                model.reset_metrics()
                model.save_weights(self.ckpt_path.format(type=model.get_name()) + f"/cp-{episode:04d}", save_format='tf')
                print(f'--- MODELLO {model.get_name()} salvato con successo ---')
            except Exception as err:
                print(f"--- Non è stato possibile salvare il MODELLO {model.get_name()} ---\n", err)

    def load_models(self, *models: DQModel):
        for model in models:
            last_ckpt = tf.train.latest_checkpoint(self.ckpt_path.format(type=model.get_name()))
            try:
                model.load_weights(last_ckpt)
                print(f"--- MODELLO {model.get_name()} (ckeckpoint: {last_ckpt}) caricato con successo ---")
            except Exception as err:
                print(f"--- Non è stato possibile caricare il MODELLO {model.get_name()} ---\n", err)

    def save_parameters(self, total_steps: int, episode: int, eps: float, session: str):
        self.parameters_helper.save_parameters(total_steps=total_steps, episode=episode, eps=eps, session=session)

    def load_parameters(self):
        return self.parameters_helper.load_parameters()
