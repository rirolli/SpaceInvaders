import json
import datetime as dt

from costants import MAX_EPSILON


class JsonHelper:
    def __init__(self, parameters_path: str):
        self.parameters_path = parameters_path

    @staticmethod
    def create_map(total_steps: int, episode: int, eps: float, session: str):
        data = {}
        data['episode'] = episode
        data['total_steps'] = total_steps
        data['eps'] = eps
        data['session'] = session

        return data

    def save_parameters(self, total_steps: int, episode: int, eps: float, session: str):
        try:
            writer = open(self.parameters_path, 'w')
            json.dump(self.create_map(total_steps, episode, eps, session), writer, indent=4)
            print('--- PARAMETRI salvati con successo ---')
            writer.close()
        except Exception as err:
            print("--- Non è stato possibile salvare i PARAMETRI ---\n", err)

    def load_parameters(self):
        try:
            reader = open(self.parameters_path, 'r')
            data = json.load(reader)
            episode = data['episode']
            total_steps = data['total_steps']
            eps = data['eps']
            session = data['session']
            reader.close()
            print("--- PARAMETRI caricati con successo ---")
            print(f"Episodio: {episode}, total_steps: {total_steps}, eps: {eps:.3f}")
        except Exception as err:
            print(f"--- Non è stato possibile caricare i PARAMETRI ---\n", err)
            episode = 0
            total_steps = 0
            eps = MAX_EPSILON
            session = dt.datetime.now().strftime('%d%m%Y%H%M')
        return episode, total_steps, eps, session


