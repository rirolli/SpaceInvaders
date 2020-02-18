import tensorflow as tf
import numpy as np

from tensorflow import keras

from costants import GAMMA, TAU, BATCH_SIZE, CKPT_PATH


class DQModel(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, str_name: str):
        super(DQModel, self).__init__()
        self.str_name = str_name.lower()
        self.conv1 = keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu')
        self.conv2 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu')
        self.flatten = keras.layers.Flatten()
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',
                                            kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions,
                                          kernel_initializer=keras.initializers.he_normal())
        self.v_dense = keras.layers.Dense(hidden_size, activation='relu',
                                          kernel_initializer=keras.initializers.he_normal())
        self.v_out = keras.layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
        self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
        self.combine = keras.layers.Add()

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.flatten(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        v = self.v_dense(x)
        v = self.v_out(v)
        norm_adv = self.lambda_layer(adv)
        combined = self.combine([v, norm_adv])
        return combined

    def update_network(self, target_network):
        # aggiorna i parametri della rete target_network lentamente con quelli della online_network
        for t, e in zip(target_network.trainable_variables, self.trainable_variables):
            t.assign(t * (1 - TAU) + e * TAU)

    def train_model(self, memory, target_network=None):
        states, actions, rewards, next_states, done = memory.sample()
        # predice Q(s,a) dato un batch di stati (state)
        prim_qt = self.call(input=states)
        # predice Q(s',a') con la online_network
        prim_qtp1 = self.call(input=next_states)
        # copia il tensore prim_qt nel tensore target_q - aggiorneremo quindi un indice corrispondente all'azione massima
        target_q = prim_qt.numpy()
        updates = rewards
        valid_idxs = done != True
        batch_idxs = np.arange(BATCH_SIZE)
        if target_network is None:
            updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
        else:
            prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
            q_from_target = target_network.call(input=next_states)
            updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
        target_q[batch_idxs, actions] = updates
        loss = self.train_on_batch(states, target_q)
        return loss

    def get_name(self):
        return self.str_name
