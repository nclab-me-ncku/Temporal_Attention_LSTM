import os
from re import L
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
TEST_START_INDEX = 5000
TAPSIZE_LIST = [10, 15, 20]
TRAIN_COUNT = 5000
TAPSIZE_THRESHOLD = 0.9
SAVE_FILE_NAME = './results/lstm_monkeyIndy_findTapSize.csv'

axis = ['x', 'y', 'z']
movementName = ['position', 'velocity', 'acceleration']
SELECT_AXIS = 1
SELECT_MOVEMENT = 1

# functions
def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

    return r_square

def get_GrandTotal(attnMap: np.ndarray):
    attnMap = attnMap.squeeze(axis=-1)
    attnMap = np.sum(attnMap, axis=0)
    attnMap = attnMap[::-1]

    sumUp = [attnMap[0]]
    for i in range(1, attnMap.shape[-1]):
        sumUp.append(attnMap[i] + sumUp[-1])
    sumUp = np.array(sumUp) / sumUp[-1]
    
    
    optTapSize = np.argmax(sumUp > TAPSIZE_THRESHOLD)
   
    return optTapSize + 1


# training function
def train(train_x, train_y, tapsize, attn):
    model = lstm_decoder(tapsize=tapsize, attn=attn)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    model.fit(x=train_x, y=train_y, batch_size=BATCHSIZE, epochs=EPOCH, verbose=0, shuffle=True)
    
    return model

# testing function
def test(model:lstm_decoder, test_x, test_y):
    if model.attn:
        pred, scale = model.predict(x=test_x)
    else:
        pred = model.predict(x=test_x)

    pred = pred.flatten()
    test_y = test_y.flatten()

    # r-square
    r2 = r_square(pred=pred, true=test_y)

    # attention scale
    if model.attn:        
        optTapSize = get_GrandTotal(scale)
        return r2, optTapSize
    else:
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
        data = data.sort_values(by=['timestamp'])
        data = data.reset_index(drop=True)
        
        #       
        select_cols = [f'firingRate_lag_{i}' for i in range(29, 0, -1)]
        select_cols = select_cols + ['firingRate']   
        
        for col in select_cols:
            data[col] = data[col].map(lambda x: np.array(x.tolist()))  
        m1 = np.array([np.array(data[col].to_list()) for col in select_cols])
        m1 = np.sum(m1, axis=-1)
        m1 = np.swapaxes(m1, 0, 1)

        movement = data[f'{movementName[SELECT_MOVEMENT]}_{axis[SELECT_AXIS]}'].to_numpy()

        # type change
        m1 = m1.astype(np.float32)
        movement = movement.astype(np.float32)

        for n , tapsize in itertools.product(range(N_COUNT), TAPSIZE_LIST):
            print('=============================')
            print('Count =>', n+1, 'Now =>', fileName, '\nINIT.TAPSZIE =>', tapsize, 'TRAINCOUNT =>', TRAIN_COUNT, \
                '\nEPOCH =>', EPOCH, 'MOVEMENT =>', movementName[SELECT_MOVEMENT], '\nAXIS =>', axis[SELECT_AXIS])

            
            # calculate z-score with training datas' parameters            
            movement = (movement - movement[:TRAIN_COUNT].mean()) / movement[:TRAIN_COUNT].std()

            # get dataset
            train_x = m1[:TRAIN_COUNT, :, :]
            train_y = movement[:TRAIN_COUNT]
            test_x = m1[TEST_START_INDEX:]
            test_y = movement[TEST_START_INDEX:]

            print('=============================')
            print('[TRAIN] with attention')

            model = train(train_x, train_y, tapsize, True)
            r2_ori, optTapSize = test(model, test_x, test_y)

            print('# find optimized tapSize =>', optTapSize)
            print('[RETRAIN] with optimized tapsize')

            model = train(train_x, train_y, optTapSize, False)
            r2_opt = test(model, test_x, test_y)

            
            df = pd.DataFrame({
                'sessionIndex': [session_index+1], 
                'sessionName': [os.path.splitext(fileName)[0]],
                'originTapSize': [tapsize],
                'optimizedTapSize': [optTapSize],
                'originRsquare': [r2_ori],
                'optimizedRsquare': [r2_opt]
            })
            df.to_csv(SAVE_FILE_NAME, index=False, header=False, mode='a')

if __name__=='__main__':
    main()