import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow import keras


class RL_Brain():
    def __init__(self, state_size, n_action):
        self.state_size = state_size
        self.n_action = n_action
        self.model = self.build_model()

    def build_model(self):
        input_ = Input(shape=(self.state_size,), name='input')
        layer1 = Dense(64, 'relu', name='encode/layer1')(input_)
        layer1 = Dropout(0.3, name='encode/dropout')(layer1)
        layer2 = Dense(64, 'relu', name='encode/layer2')(layer1)
        layer3 = Dense(64, 'relu', name='encode/layer3')(layer2)
        fai = Dense(512, 'relu', name='fai')(layer3)
        decoder1 = Dense(64, 'relu', name='decode/layer1')(fai)
        decoder2 = Dense(64, 'relu', name='decode/layer2')(decoder1)
        decoder3 = Dense(64, 'relu', name='decode/layer3')(decoder2)
        s_hat = Dense(self.state_size, name='output_s_hat')(decoder3)
        R = Dense(1, name='R', use_bias=False)(fai)
        mus = []
        for i in range(self.n_action):
            mu = Dense(512, 'relu', name='mu/action%s/layer1' % i)(fai)
            mu = Dense(256, 'relu', name='mu/action%s/layer2' % i)(mu)
            mu = Dense(512, 'relu', name='mu/action%s/layer3' % i)(mu)
            mus.append(mu)

        model = Model(inputs=input_, outputs=[R, s_hat, mus])
        return model

    @tf.function
    def traceme(self, x):
        return self.model(x)

    def visualize_model(self):
        log_dir = './log'
        writer = tf.summary.create_file_writer(log_dir)
        tf.summary.trace_on(graph=True)
        self.traceme(tf.zeros(shape=(5, 2)))
        with writer.as_default():
            tf.summary.trace_export('model_trace', step=0)
        keras.utils.plot_model(self.model, './dsr_model.png', True, True)


if __name__ == '__main__':
    brain = RL_Brain(2, 4)
    keras.utils.plot_model(brain.model, './dsr_model.png', show_shapes=True)
    brain.visualize_model()
