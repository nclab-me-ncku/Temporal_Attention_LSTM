import os

from tensorflow.python.keras.layers.normalization.layer_normalization import LayerNormalization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from einops import repeat
import itertools


# PARAM
BATCHSIZE = 64
EPOCH = 60

N_COUNT = 5

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

def train(train_x, train_y):
    model = lstm_decoder(tapsize=30)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    model.fit(x=train_x, y=train_y, batch_size=BATCHSIZE, epochs=EPOCH, verbose=0, shuffle=True)

    return model

def test(model:keras.Model, test_x, test_y):
    pred = model.predict(x=test_x)
    pred = pred.flatten()
    test_y = test_y.flatten()

    def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
        ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
        ss_res = np.sum((true - pred) ** 2, axis=0)
        r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

        return r_square

    r2 = r_square(pred=pred, true=test_y)

    return r2

def main():
    # LOAD DATASET
    folderPath = './data/indy'
    fileList = np.array(sorted([f for f in os.listdir(folderPath) if f.endswith('npz')]))
    selected_session = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 21, 22, 23, 28, 29, 31, 36]

    for TAPSIZE, TRAIN_COUNT, n in itertools.product([5, 20, 5, 10, 15], [3000, 5000, 7000], range(N_COUNT)):
        for session_index, fileName in enumerate(fileList[selected_session]):
            print('=============================')
            print('Count =>', n+1, 'Now =>', fileName, '\nTAPSZIE =>', TAPSIZE, 'TRAINCOUNT =>', TRAIN_COUNT)

            filePath = os.path.join(folderPath, fileName)
            

            data = np.load(filePath)

            m1, movement = data['m1'], data['movement']

            m1 = m1.astype(np.float32)
            movement = movement.astype(np.float32)

            s_ax = SELECT_MOVEMENT * 3 + SELECT_AXIS
            X = m1[:, :, :, :][:TRAIN_COUNT]
            Y = movement[:, s_ax:s_ax + 1][:TRAIN_COUNT]
            A = m1[:, :, :, :][7000:]
            B = movement[:, s_ax:s_ax + 1][7000:]

            results = []

            # Channel base
            print('=============================')
            train_x = np.sum(X, axis=3)
            train_y = Y
            test_x = np.sum(A, axis=3)
            test_y = B
            model = train(train_x, train_y)
            r2 = test(model, test_x, test_y)
            results.append(
                [os.path.splitext(fileName)[0], 'channelbase', r2, axis[SELECT_AXIS], \
                    movementName[SELECT_MOVEMENT], TAPSIZE, TRAIN_COUNT])

            df = pd.DataFrame(results, columns=['session', 'method', 'r2', 'axis', 'movement', 'timeLag', 'trainCount'])
            df.to_csv('./LSTM_MONKEY_N.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    main()