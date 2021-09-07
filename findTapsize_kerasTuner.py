import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import tensorflow.keras as keras
import pandas as pd
from einops import repeat
import itertools


# PARAM
BATCHSIZE = 64
EPOCH = 60

axis = ['x', 'y', 'z']
movementName = ['position', 'velocity', 'acceleration']
SELECT_AXIS = 0
SELECT_MOVEMENT = 1



# MODEL
class TAM(keras.Model):
    def __init__(self, featureNum, reduction_ratio):
        super(TAM, self).__init__()
        
        self.scoreLayer = keras.Sequential([
            keras.layers.Dense(featureNum//reduction_ratio, activation=None),
            keras.layers.LayerNormalization(),
            keras.layers.Activation(tf.nn.tanh),
            keras.layers.Dense(1, activation=None),
            keras.layers.Flatten()            
        ])

        self.softmax = keras.layers.Softmax(axis=-1)

    def call(self, x):        

        score = self.scoreLayer(x)

        # seqlen = tf.cast(x.shape[1], tf.float32)       
        # score = score / tf.sqrt(seqlen)

        score = self.softmax(score)
        score = repeat(score, 'bs t -> bs t a', a=1)
        x = tf.multiply(x, tf.broadcast_to(score, tf.shape(x)))
        x = tf.reduce_sum(x, axis=1, keepdims=True)

        return x, score

class lstm_decoder(keras.Model):
    def __init__(self, tapsize):
        super(lstm_decoder, self).__init__()

        # params
        self.tapsize = tapsize

        # layers
        self.emb = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True), merge_mode='concat'),
            keras.layers.LayerNormalization(),
            keras.layers.LSTM(256, return_sequences=True)
        ])   

        self.tmpAttn = TAM(featureNum=256, reduction_ratio=2)

        self.decoder = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

    def call(self, x):
        x = x[:, -self.tapsize:, :]
        x = self.emb(x)   
        x, scale = self.tmpAttn(x)
        x = self.decoder(x)
        
        return x

def build_model(hp):
    model = lstm_decoder(tapsize=hp.Choice('timeLag', [i+1 for i in range(30)]))    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    return model


def main():
    # LOAD DATASET
    folderPath = './data/indy'
    fileList = np.array(sorted([f for f in os.listdir(folderPath) if f.endswith('npz')]))

    
    for session_index, fileName in enumerate(fileList[:1]):
        print('=============================')

        filePath = os.path.join(folderPath, fileName)
        
        data = np.load(filePath)

        m1, movement = data['m1'], data['movement']

        m1 = m1.astype(np.float32)
        movement = movement.astype(np.float32)

        s_ax = SELECT_MOVEMENT * 3 + SELECT_AXIS
        X = m1[:, :, :, :][:5000]
        Y = movement[:, s_ax:s_ax + 1][:5000]
        A = m1[:, :, :, :][5000:]
        B = movement[:, s_ax:s_ax + 1][5000:]

        train_x = np.sum(X, axis=3)
        train_y = Y
        test_x = np.sum(A, axis=3)
        test_y = B


        #
        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=5)
        
        tuner.search(train_x, train_y, epochs=5, validation_data=(test_x, test_y))

        best_model = tuner.get_best_models()[0]
       
 
      
       

if __name__ == '__main__':
    main()