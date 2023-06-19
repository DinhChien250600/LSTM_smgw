import numpy as np
import pandas
import time

from tflite_runtime.interpreter import Interpreter
from sklearn.preprocessing import MinMaxScaler

import pathlib

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
    #inputs_scale = inputs_scale.reshape(1,60,1)
    
    # Load TFLite model and allocate tensors.
    #interpreter = Interpreter(model_path="./LSTM_model/LSTM_model_6_temp.tflite")
    interpreter = Interpreter(model_path="./LSTM_model/LSTM_model_2_temp_fl16.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    output = np.random.randn(2916, 1)
    count=0
    for i in range(2916):
        input_test = inputs[i].reshape(1,60,1)
        input_test_32 = np.float32(input_test)
        interpreter.set_tensor(input_details[0]['index'], input_test_32)
        interpreter.invoke()
        output_test = interpreter.get_tensor(output_details[0]['index'])
        output[count] = output_test
        count = count + 1
    
    output_inverse = scaler.inverse_transform(output)
    
    return output_inverse

start_time = time.time()
y_pred_RNN_model = predict_params(X_test)
end_time = time.time()
elapsed_time = end_time - start_time
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
