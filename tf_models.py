from keras.models import Model
from keras.layers import Input
from keras.layers.core import *
from keras.layers.recurrent import *
from tcn import TCN

import tensorflow as tf
from keras import backend as K


# Custom loss from kidzinski et al.
def custom_loss(weight=0.01):
    def weighted_binary_crossentropy(y_true, y_pred):
        a1 = K.mean(np.multiply(K.binary_crossentropy(y_pred, y_true),(y_true + weight)), axis=-1)
        return a1
    return weighted_binary_crossentropy


#  ------------------------ Models -------------------------------------
def uniDirLSTM(n_nodes, drop, n_classes, n_feat, n_layers, weight, max_len,
              causal=True, loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer()):

    inputs = Input(shape=(None, n_feat))
    x = LSTM(n_nodes, return_sequences=True)(inputs)

    if n_layers == 2:
        x = LSTM(n_nodes, return_sequences=True)(x)

    model = Dense(n_classes, activation="sigmoid")(x)

    model = Model(input=inputs, output=model)
    model.compile(optimizer=optimizer, loss=custom_loss(weight), metrics=['accuracy'])

    return model


def TCN_keras(n_nodes, drop, n_classes, n_feat, n_kernels, weight, max_len, dilations,
        causal=True, loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer()):

    if causal:
        padding = 'causal'
    else:
        padding = 'same'

    input_layer = Input(shape=(None, n_feat))

    tcn1 = TCN(nb_filters=n_nodes, kernel_size=n_kernels, nb_stacks=1, return_sequences=True,
               dropout_rate=drop, activation='relu', use_batch_norm=True, use_skip_connections=False, dilations=dilations, padding=padding)(input_layer)

    output_layer = Dense(n_classes, activation='sigmoid')(tcn1)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=custom_loss(weight), optimizer=optimizer, metrics=['accuracy'])

    return model




