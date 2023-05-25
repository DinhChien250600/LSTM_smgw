import numpy as np
import pandas
import time

from tensorflow.keras.models import load_model
from tensorflow.python.ops.math_ops import reduce_prod
from sklearn.preprocessing import MinMaxScaler

#test_2019_1 = pandas.read_csv('./2019/1.csv', usecols=['Date', 'Hour', 'Temperature Number'])
#test_2019_2 =  pandas.read_csv('./2019/2.csv', usecols=['Date', 'Hour', 'Temperature Number'])
#test_2019_3 =  pandas.read_csv('./2019/3.csv', usecols=['Date', 'Hour', 'Temperature Number'])

#data_test = pandas.concat([test_2019_1,test_2019_2,test_2019_3[:144]])
#data_test[60:]

#scaler = MinMaxScaler()

#df = data_test.drop(['Date', 'Hour'], axis = 1)

#inputs = scaler.fit_transform(df)

#X_test = []
#y_test = []

#for i in range(60, inputs.shape[0]):
#    X_test.append(inputs[i-60:i])
#    y_test.append(inputs[i, 0])
    
#X_test, y_test = np.array(X_test), np.array(y_test)
#print(X_test.shape)
#print(y_test.shape)

def predict_params(input,a):
    data_test = pandas.read_csv('./2019/1.csv', usecols=['Date', 'Hour', 'Temperature Number'])
    df = data_test.drop(['Date', 'Hour'], axis = 1)
    inputs = df[0:59].values.tolist()
    inputs.append([input])
    
    scaler = MinMaxScaler()
    inputs_scale = scaler.fit_transform(inputs)
    
    inputs_scale = inputs_scale.reshape(1,60,1)
    RNN_model = load_model('RNN_model_temp.h5', compile=False)

    print("Model is predict...")
    y_pred_RNN_model = RNN_model.predict(inputs_scale)
    y_pred_RNN_model = scaler.inverse_transform(y_pred_RNN_model)
    
    return y_pred_RNN_model.item()

print("Predict with RNN model")
start_time = time.time()
y_predict = predict_params(30,26)
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)
print("Predict: ",y_predict)
#y_pred_RNN_model = scaler.inverse_transform(y_pred_RNN_model)
#y_test_new = scaler.inverse_transform(y_test.reshape(-1, 1))
#print(y_pred_RNN_model[0])
#print(y_test_new[0])

