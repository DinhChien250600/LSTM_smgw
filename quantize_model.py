import tensorflow as tf
from tensorflow.keras.models import load_model
import pathlib

#RNN_model = load_model('./LSTM_model/LSTM_model_2_temp.h5', compile=False)
RNN_model = load_model('RNN_model_temp.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(RNN_model)
RNN_model_tflite = pathlib.Path("/home/sanslab/test/RNN_model_temp_3.tflite")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]
#converter.experimental_new_quantizer = True
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.experimental_new_converter=True
#converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
RNN_model_tflite.write_bytes(tflite_quant_model)

