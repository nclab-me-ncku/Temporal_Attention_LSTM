# Temporal_Attention_LSTM

## Introduction and environment

> * This repository implements temporal attention-aware timestep selection (TTS) method for LSTM. 
> * Our research has exerted this technique in neural decoding. Experimental results show that it could **outperform state-of-the-art neural decoders** on two nonhuman primate datasets. In addition, it also **reduces the computation time for prediction**.
> * Here is our environment: 
>> * OS: Windows 10
>>* Language: python 3.8.5
>>* Packages: includes in `Pipfile`

## Why we need timestep selection

RNN-based neural decoders might cause latency between the input of neural activity and the response of kinematic state because of insufficient number of timesteps (time bins) signal.

Therefore, adding both previous and current timesteps signal could help the model learn neural response dynamics from neural activity efficiently. 

However, excessively long neural activity periods results in computational burden and hinders the decoding performance.  

Accordingly, it is important to make a trade-off between the computational complexity of decoder and the decoding performance by selecting adequate number of input timesteps.

## What is temporal attention module (TAM)

The TAM in ours research aims to determine the relative importance of each neural activity timestep and selects essential timesteps by means of attention weight <img src="https://latex.codecogs.com/gif.latex?a_{\tau}"/> 



## How to use our model

 

