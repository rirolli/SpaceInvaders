import tensorflow as tf


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate):
        # Input
        self.input = tf.placeholder(tf.float32, [None, *state_size],name="input")
        self.conv1 = tf.layers.conv2d(inputs=self.input,
                                      filters=32,
                                      kernel_size=[8, 8],
                                      strides=[4,4],
                                      padding="VALID",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv1")

        self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

        self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                      filters=64,
                                      kernel_size=[4, 4],
                                      strides=[2, 2],
                                      padding="VALID",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv2")

        self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

        self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                      filters=64,
                                      kernel_size=[3, 3],
                                      strides=[2, 2],
                                      padding="VALID",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv3")

        self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

        self.flatten = tf.contrib.layers.flatten(self.conv3_out)

        self.fc = tf.layers.dense(inputs=self.flatten,
                                  units=512,
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")

        self.output = tf.layers.dense(inputs=self.fc,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      units=action_size,
                                      activation=None)

        # self.action_chosen = tf.argmax(self.output, 1)
        #---

        self.actions = tf.placeholder(tf.float32, [None, action_size], name="actions")

        # Q is our predicted Q value.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))

        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None], name="target")
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.trainer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)