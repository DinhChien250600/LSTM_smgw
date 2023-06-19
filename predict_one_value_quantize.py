import numpy as np
from tflite_runtime.interpreter import Interpreter
import pandas
from sklearn.preprocessing import MinMaxScaler
import time

def predict_params(input,a):
    data_test = pandas.read_csv('/home/sanslab/test/2019/1.csv', usecols=['Date', 'Hour', 'Temperature Number'])
    df = data_test.drop(['Date', 'Hour'], axis = 1)
    inputs = df[0:59].values.tolist()
    inputs.append([input])
    
    scaler = MinMaxScaler()
    inputs_scale = scaler.fit_transform(inputs)
    
    inputs_scale = inputs_scale.reshape(1,60,1)
    
    # Load TFLite model and allocate tensors.
    interpreter = Interpreter(model_path="./LSTM_model/LSTM_model_2_temp.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    inputs_scale_float32 = np.float32(inputs_scale)
    
    interpreter.set_tensor(input_details[0]['index'], inputs_scale_float32)
    
    interpreter.invoke()
    
    print("Model is predict...")

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    #y_pred_RNN_model = RNN_model.predict(inputs_scale)
    y_pred_RNN_model = scaler.inverse_transform(output_data)
    
    return y_pred_RNN_model.item()
    
start_time = time.time()
y_pred = predict_params(26,1)
end_time = time.time()
elapsed_time = end_time - start_time
print("Data predict:", y_pred)
print("Time for predict: ", elapsed_time)
