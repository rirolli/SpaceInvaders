from model import DQModel

from json_helper import JsonHelper


class Saver:
    def __init__(self, ckpt_path: str, parameters_path: str):
        self.ckpt_path = ckpt_path
        self.parameters_path = parameters_path
        self.parameters_helper = JsonHelper(self.parameters_path)

    def save_models(self, episode: int, *models: DQModel):
        for model in models:
            try:
                model.save_weights(self.ckpt_path.format(type=model.get_name(), epoch=episode))
                print(f'--- MODELLO {model.get_name()} salvato con successo ---')
            except Exception as err:
                print(f"--- Non è stato possibile salvare il MODELLO {model.get_name()} ---\n", err)

    def load_models(self, *models: DQModel):
        last_episode, _, _, _ = self.parameters_helper.load_parameters()
        for model in models:
            try:
                model.load_weights(self.ckpt_path.format(type=model.get_name(), epoch=last_episode))
                print(f"--- MODELLO {model.get_name()} caricato con successo ---")
            except Exception as err:
                print(f"--- Non è stato possibile caricare il MODELLO {model.get_name()} ---\n", err)

    def save_parameters(self, total_steps: int, episode: int, eps: float, session: str):
        self.parameters_helper.save_parameters(total_steps=total_steps, episode=episode, eps=eps, session=session)

    def load_parameters(self):
        return self.parameters_helper.load_parameters()
