import numpy as np
import pandas
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict_params(input,a):
    data_test = pandas.read_csv('/home/sanslab/test/2019/1.csv', usecols=['Date', 'Hour', 'Temperature Number'])
    df = data_test.drop(['Date', 'Hour'], axis = 1)
    inputs = df[0:59].values.tolist()
    inputs.append([input])
    
    scaler = MinMaxScaler()
    inputs_scale = scaler.fit_transform(inputs)
    
    inputs_scale = inputs_scale.reshape(1,60,1)
    
    RNN_model = load_model('./LSTM_model/LSTM_model_4_temp.h5', compile=False)
    
    print("Model is predict...")
    
    y_pred_RNN_model = RNN_model.predict(inputs_scale)
    
    y_pred_RNN_model_inverse = scaler.inverse_transform(y_pred_RNN_model)
    
    return y_pred_RNN_model_inverse.item()

start_time = time.time()
y_pred = predict_params(26,1)
end_time = time.time()
elapsed_time = end_time - start_time
print("Data predict:", y_pred)
print("Time for predict: ", elapsed_time)
