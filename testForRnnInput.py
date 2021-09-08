'''
This file is for testing the number of RNN's input. 
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import itertools
from models import lstm_decoder

# PARAM
BATCHSIZE = 64
EPOCH = 60
N_COUNT = 5
TEST_START_INDEX = 7000
TAPSIZE_LIST = [10, 15, 20]
TRAIN_COUNT_LIST = [5000, 1178, 1325]
SAVE_FILE_NAME = './results/lstm_monkeyIndy.csv'

axis = ['x', 'y', 'z']
movementName = ['position', 'velocity', 'acceleration']
SELECT_AXIS = 0
SELECT_MOVEMENT = 1


def train(train_x, train_y, tapsize):
    model = lstm_decoder(tapsize=tapsize, attn=True)
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
    fileList = np.array(sorted([f for f in os.listdir(folderPath) if f.endswith('feather')]))

    
    for session_index, fileName in enumerate(fileList):
        
        filePath = os.path.join(folderPath, fileName)
            
        data = pd.read_feather(filePath)

        # setting filter
        data = data.dropna(axis=0)
        data = data[data['sensor'] == 'm1']
        # data = data[data['moved'] == True]            
        data = data.sort_values(by=['timestamp'])
        data = data.reset_index(drop=True)
        
        #
        select_cols = [f'firingRate_lag_{i}' for i in range(29, 0, -1)]
        select_cols = select_cols + ['firingRate']                           
        m1 = data[select_cols]
        m1 = m1.applymap(lambda x: np.vstack(x))
        m1 = m1.to_numpy()
        
        m1 = np.array([[m1[i, j] for j in range(30)] for i in range(len(m1))])
        m1 = np.sum(m1, axis=-1)

        movement = data[f'{movementName[SELECT_MOVEMENT]}_{axis[SELECT_AXIS]}'].to_numpy()

        # type change
        m1 = m1.astype(np.float32)
        movement = movement.astype(np.float32)

        
        for n , tapsize, trainCount in itertools.product(range(N_COUNT), TAPSIZE_LIST, TRAIN_COUNT_LIST):
            print('=============================')
            print('Count =>', n+1, 'Now =>', fileName, '\nTAPSZIE =>', tapsize, 'TRAINCOUNT =>', trainCount)

            
            # z-score            
            movement = (movement - movement[:trainCount].mean()) / movement[:trainCount].std()

            print('=============================')
            train_x = m1[:trainCount, :, :]
            train_y = movement[:trainCount]
            test_x = m1[TEST_START_INDEX:]
            test_y = movement[TEST_START_INDEX:]
            
            model = train(train_x, train_y, tapsize)
            r2 = test(model, test_x, test_y)
            results = [os.path.splitext(fileName)[0], 'channelbase', r2, axis[SELECT_AXIS], \
                    movementName[SELECT_MOVEMENT], tapsize, trainCount]

            df = pd.DataFrame([results], columns=['session', 'method', 'r2', 'axis', 'movement', 'timeLag', 'trainCount'])
            df.to_csv(SAVE_FILE_NAME, mode='a', header=False, index=False)

            

if __name__ == '__main__':
    main()