import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import itertools
import time
import argparse
from models import bciDecoder


# functions
def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

    return r_square

def get_GrandTotal(attnMap: np.ndarray):
    global TAPSIZE_THRESHOLD

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
    global EPOCH_COUNT
    model = bciDecoder(tapsize=tapsize, attn=attn)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    model.fit(x=train_x, y=train_y, batch_size=BATCHSIZE, epochs=EPOCH_COUNT, verbose=0, shuffle=True)
    
    return model

# testing function
def test(model:bciDecoder, test_x, test_y):
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
    global DATA_FOLDER, BATCHSIZE, EPOCH_COUNT, N_COUNT, TAPSIZE_INIT, TRAIN_COUNT, SAVE_FILE_NAME, MOVEMENT_NAME

    
    # LOAD DATASET
    fileList = np.array(sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('feather')]))

    for session_index, fileName in enumerate(fileList):
        
        filePath = os.path.join(DATA_FOLDER, fileName)

        print(f'[PRE] reading dataset from {filePath}')
        data = pd.read_feather(filePath)


        # setting filter                       
        data = data.sort_values(by=['timestamp'])
        data = data.reset_index(drop=True)

        # tapsize      
        print('[PRE] stack up training datas')
        data['firingRate'] = data['firingRate'].map(lambda x: np.array(x.tolist())) 
        for shift_time in range(1, TAPSIZE_INIT):
            data[f'firingRate_lag_{shift_time}'] = data['firingRate'].shift(shift_time)
        data = data.dropna(axis=0)

        #       
        select_cols = [f'firingRate_lag_{i}' for i in range(TAPSIZE_INIT-1, 0, -1)]
        select_cols = select_cols + ['firingRate']   
        
        m1 = np.array([np.array(data[col].to_list()) for col in select_cols])
        m1 = np.sum(m1, axis=-1)
        m1 = np.swapaxes(m1, 0, 1)

        movement = data[MOVEMENT_NAME].to_numpy()

        # type change
        m1 = m1.astype(np.float32)
        movement = movement.astype(np.float32)

        for n , tapsize in itertools.product(range(N_COUNT), [TAPSIZE_INIT]):
            print('=============================')
            print('[Info] Count =>', n+1, 'Now =>', fileName)

            
            # calculate z-score with training datas' parameters            
            movement = (movement - movement[:TRAIN_COUNT].mean()) / movement[:TRAIN_COUNT].std()

            # get dataset
            train_x = m1[:TRAIN_COUNT, :, :]
            train_y = movement[:TRAIN_COUNT]
            test_x = m1[TRAIN_COUNT:]
            test_y = movement[TRAIN_COUNT:]

            
            print('[TRAIN] with attention')
            
            searchStartTime = time.time()
            model = train(train_x, train_y, tapsize, True)
            r2_ori, optTapSize = test(model, test_x, test_y)
            searchEndTime = time.time()

            print('[Info] find optimized tapSize =>', optTapSize)
            print('[RETRAIN] with optimized tapsize')
            
            trainStartTime = time.time()
            model = train(train_x, train_y, optTapSize, False)
            trainEndTime = time.time()
            
            testStartTime = time.time()
            r2_opt = test(model, test_x, test_y)
            testEndTime = time.time()
            
            df = pd.DataFrame({
                'sessionIndex': [session_index+1], 
                'sessionName': [os.path.splitext(fileName)[0]],
                'originTapSize': [tapsize],
                'optimizedTapSize': [optTapSize],
                'originRsquare': [r2_ori],
                'optimizedRsquare': [r2_opt],
                'searchUsedTime': [searchEndTime - searchStartTime],
                'trainUsedTime': [trainEndTime - trainStartTime],
                'testUsedTime': [testEndTime - testStartTime],
            })
            df.to_csv(SAVE_FILE_NAME, index=False, header=False, mode='a')
    
    df = pd.read_csv(SAVE_FILE_NAME, header=None)
    df.columns = ['sessionIndex', 'sessionName', 'originTapSize', 'optimizedTapSize', 'originRsquare', \
        'optimizedRsquare', 'searchUsedTime', 'trainUsedTime', 'testUsedTime']
    df.to_csv(SAVE_FILE_NAME, index=False, mode='w')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFolder', default='./data', type=str)
    parser.add_argument('--outputFolder', default='./results', type=str)
    parser.add_argument('--movementName', default='velocity_y', type=str)
    parser.add_argument('--batchSize', default=64, type=int)
    parser.add_argument('--epochCount', default=60, type=int)
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument('--initTapsize', default=20, type=int)
    parser.add_argument('--runCount', default=5, type=int)
    parser.add_argument('--trainingSampleCount', default=5000, type=int)
    args = parser.parse_args()

    #
    BATCHSIZE = args.batchSize
    EPOCH_COUNT = args.epochCount
    N_COUNT = args.runCount
    TAPSIZE_INIT = args.initTapsize
    TRAIN_COUNT = args.trainingSampleCount
    TAPSIZE_THRESHOLD = args.threshold
    SAVE_FILE_NAME = os.path.join(args.outputFolder, 'tts_results.csv')
    DATA_FOLDER = args.inputFolder
    MOVEMENT_NAME = args.movementName

    #
    main()