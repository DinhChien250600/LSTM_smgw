import numpy as np
import pandas
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

test_2019_1 = pandas.read_csv('./2019/1.csv', usecols=['Date', 'Hour', 'Temperature Number'])
test_2019_2 =  pandas.read_csv('./2019/2.csv', usecols=['Date', 'Hour', 'Temperature Number'])
test_2019_3 =  pandas.read_csv('./2019/3.csv', usecols=['Date', 'Hour', 'Temperature Number'])

data_test = pandas.concat([test_2019_1,test_2019_2,test_2019_3[:144]])
data_test[60:]

scaler = MinMaxScaler()

df = data_test.drop(['Date', 'Hour'], axis = 1)

inputs = scaler.fit_transform(df)

X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)

def predict_params(inputs):
    RNN_model = load_model('./LSTM_model/LSTM_model_6_temp.h5', compile=False)

    print("Model is predicting...")
    
    y_pred_RNN_model = RNN_model.predict(X_test)
    
    return y_pred_RNN_model

start_time = time.time()
y_pred_RNN_model = predict_params(inputs)
end_time = time.time()
elapsed_time = end_time - start_time
y_pred_RNN_model = scaler.inverse_transform(y_pred_RNN_model)
y_test_new = scaler.inverse_transform(y_test.reshape(-1, 1))

lose_in_model_1=abs(y_test_new-y_pred_RNN_model)

mean = np.mean(lose_in_model_1)
min = np.amin(lose_in_model_1) 
max = np.amax(lose_in_model_1) 
range = np.ptp(lose_in_model_1) 
varience = np.var(lose_in_model_1) 
sd = np.std(lose_in_model_1)

print("Time for predict: ", elapsed_time)
print("Measures of Dispersion")
print("Mean=", mean)
print("Minimum =", min) 
print("Maximum =", max) 
print("Range =", range) 
print("Varience =", varience) 
print("Standard Deviation =", sd) 