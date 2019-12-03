import time
import random
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from collections import deque

from variables import tensorboard_path, model_path, load_models_path, load_model


class DQNAgent:
    def __init__(self, state_size, action_space, learning_rate, gamma, max_memory, double_q):
        self.state_size = state_size
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.sess = tf.Session()
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=10)
        if load_model:
            self.load_model()
        else:
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.memory = deque(maxlen=max_memory)
        self.eps = 1
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.gamma = gamma
        self.batch_size = 32
        self.burning = 1000
        self.avviso_tensorboard = True
        self.copy = 1000
        self.step = 0
        self.learn_each = 50
        self.learn_step = 0
        self.save_each = 5000
        self.double_q = double_q

    def build_model(self):
        """ crea il modello della rete neurale """
        # Input -> placeholder di dimensioni state_size = (84, 84, 4)
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_size, name="input")
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")    # a true
        self.target_Q = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")   # target_q o q_true
        self.reward = tf.placeholder(dtype=tf.float32, shape=[], name='reward')
        self.input_float = tf.to_float(self.input) / 255.

        with tf.compat.v1.variable_scope('online'):
            # primo layer di convoluzione 2D
            self.conv1 = tf.layers.conv2d(inputs=self.input,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1",
                                          activation=tf.nn.relu)

            # secondo layer di convoluzione 2D
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2",
                                          activation=tf.nn.relu)

            # terzo layer di convoluzione 2D
            self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3",
                                          activation=tf.nn.relu)

            # layer di flattern (trasforma i risultati in un array)
            self.flatten = tf.contrib.layers.flatten(self.conv3)

            # primo layer Dense (outputs = activation(inputs * kernel + bias)
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            # output = tutti i valori di Q calcolati.
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_space,
                                          activation=None)

        with tf.compat.v1.variable_scope('target'):
            # primo layer di convoluzione 2D
            self.conv1_target = tf.layers.conv2d(inputs=self.input,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1",
                                          activation=tf.nn.relu)

            # secondo layer di convoluzione 2D
            self.conv2_target = tf.layers.conv2d(inputs=self.conv1_target,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2",
                                          activation=tf.nn.relu)

            # terzo layer di convoluzione 2D
            self.conv3_target = tf.layers.conv2d(inputs=self.conv2_target,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3",
                                          activation=tf.nn.relu)

            # layer di flattern (trasforma i risultati in un array)
            self.flatten_target = tf.contrib.layers.flatten(self.conv3_target)

            # primo layer Dense (outputs = activation(inputs * kernel + bias)
            self.fc_target = tf.layers.dense(inputs=self.flatten_target,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            # output = tutti i valori di Q calcolati.
            self.output_target = tf.stop_gradient(tf.layers.dense(inputs=self.fc_target,
                                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                                  units=self.action_space,
                                                                  activation=None))

            # - Ottimizzazione
            # Q è la nostra predicted Q.
            self.action = tf.argmax(input=self.output, axis=1)
            # self.q_pred = tf.reduce_sum(tf.multiply(self.output, self.actions))
            # self.loss = tf.reduce_mean(tf.square(self.target_Q - self.q_pred))
            self.q_pred = tf.gather_nd(params=self.output,
                                       indices=tf.stack([tf.range(tf.shape(self.actions)[0]), self.actions], axis=1))
            self.loss = tf.losses.huber_loss(labels=self.target_Q, predictions=self.q_pred)
            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # - Sommario
            self.writer = tf.summary.FileWriter(logdir=tensorboard_path)

            self.summaries = tf.summary.merge([
                tf.summary.scalar('reward', self.reward),
                tf.summary.scalar('loss', self.loss),
                tf.summary.scalar('max_q', tf.reduce_max(self.output))
            ])

    def load_model(self):
        """ carica i modelli delle reti """
        print (' -- Caricamento Modello... -- ')
        try:
            checkpoint = tf.train.latest_checkpoint(load_models_path)
            self.saver.restore(self.sess, checkpoint)
            print(" -- Modello {} caricato. -- ".format(checkpoint))
        except Exception:
            print(" -- Modello non trovato... -- ")
            self.sess.run(tf.global_variables_initializer())


    def copy_model(self):
        """ Copia i pesi (weights) nella target network """
        self.sess.run([tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('target'), tf.trainable_variables('online'))])

    def save_model(self):
        """ Salva il modello corrente sul disco """
        self.saver.save(sess=self.sess, save_path=model_path, global_step=self.step)
        print(" -- modello salvato con successo allo step {}! -- ".format(self.step))

    def add(self, experience):
        self.memory.append(experience)

    def predict(self, model, state):
        """ Prediction """
        if model == 'online':
            return self.sess.run(fetches=self.output, feed_dict={self.input: state})
        if model == 'target':
            return self.sess.run(fetches=self.output_target, feed_dict={self.input: state})

    def run(self, state):
        if np.random.rand() < self.eps:
            # Random action
            action = np.random.randint(low=0, high=self.action_space)
        else:
            # Policy action
            q = self.predict('online', np.expand_dims(state, 0))
            action = np.argmax(q)
        # Decrease eps
        self.eps *= self.eps_decay
        self.eps = max(self.eps_min, self.eps)
        # Increment step
        self.step += 1
        return action

    def learn(self):
        """ Gradient descent """
        # Sync target network
        if self.step % self.copy == 0:
            self.copy_model()
        # Checkpoint model
        if self.step % self.save_each == 0:
            self.save_model()
        # Break if burn-in
        if self.step < self.burning:
            return
        # Break if no training
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return

        # Sample batch
        if self.avviso_tensorboard:
            print(" - Inizio creazione modello TENSORBOARD - ")
            self.avviso_tensorboard = False
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        # Get next q values from target network
        next_q = self.predict('target', next_state)
        # Calculate discounted future reward
        if self.double_q:
            q = self.predict('online', next_state)
            a = np.argmax(q, axis=1)
            target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
        else:
            target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)
        # Update model
        summary, _ = self.sess.run(fetches=[self.summaries, self.train],
                                   feed_dict={self.input: state,
                                              self.target_Q: np.array(target_q),
                                              self.actions: np.array(action),
                                              self.reward: np.mean(reward)})
        # Reset learn step
        self.learn_step = 0
        # Write
        self.writer.add_summary(summary, self.step)
        self.writer.flush()

    '''
    def replay(self, env, model_path, n_replay, plot):
        """ replay del modello """
        ckpt = tf.train.latest_checkpoint(model_path)
        saver = tf.train.import_meta_graph(ckpt + '.meta')
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('online/output/BiasAdd:0')
        # Replay RL agent
        state = env.reset()
        total_reward = 0
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            for _ in range(n_replay):
                step = 0
                while True:
                    time.sleep(0.05)
                    env.render()
                    # Plot
                    if plot:
                        if step % 100 == 0:
                            self.visualize_layer(session=sess, layer=self.conv2, state=state, step=step)
                    # Action
                    if np.random.rand() < 0.0:
                        action = np.random.randint(low=0, high=self.actions, size=1)[0]
                    else:
                        q = sess.run(fetches=output, feed_dict={input: np.expand_dims(state, 0)})
                        action = np.argmax(q)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    state = next_state
                    step += 1
                    if info['flag_get']:
                        break
                    if done:
                        break
        env.close()

    
    def visualize_layer(self, session, layer, state, step):
        """ Visualizzazione dei layer convoluzionali """
        units = session.run(layer, feed_dict={self.input: np.expand_dims(state, 0)})
        filters = units.shape[3]
        plt.figure(1, figsize=(40, 40))
        n_columns = 8
        n_rows = np.ceil(filters / n_columns)
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0, :, :, i], interpolation="nearest", cmap='YlGnBu')
        plt.savefig(fname='./img/img-' + str(step) + '.png')
    '''
