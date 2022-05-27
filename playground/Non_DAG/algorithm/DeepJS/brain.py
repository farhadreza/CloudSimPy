import tensorflow as tf


class MyBrain(tf.keras.Model):
    name = 'MyBrain'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_6 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        state = self.dense_6(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

class BrainBig(tf.keras.Model):
    name = 'BrainBig'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_6 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        state = self.dense_6(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class Brain(tf.keras.Model):
    name = 'Brain'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        state = self.dense_5(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)

# class Brain(tf.keras.Model):
#     name = 'Brain'
#
#     def __init__(self, state_size):
#         super().__init__()
#         self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
#         self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
#         self.dense_3 = tf.keras.layers.Dense(18, activation=tf.tanh)
#         self.lstm_1 = tf.keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1])
#         self.lstm_2 = tf.keras.layers.LSTM(20, return_sequences=True)
#         self.dense_4 = tf.keras.layers.Dense(9, activation=tf.tanh)
#         self.dense_5 = tf.keras.layers.Dense(1)
#
#     def call(self, state):
#         state = self.dense_1(state)
#         state = self.dense_2(state)
#         state = self.dense_3(state)
#         state = self.lstm_1(state)
#         state = self.lstm_2(state)
#         state = self.dense_4(state)
#         state = self.dense_5(state)
#         return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)


class BrainSmall(tf.keras.Model):
    name = 'BrainSmall'

    def __init__(self, state_size):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(3, input_shape=(None, state_size), activation=tf.tanh)
        self.dense_2 = tf.keras.layers.Dense(9, activation=tf.tanh)
        self.dense_3 = tf.keras.layers.Dense(6, activation=tf.tanh)
        self.dense_4 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)
