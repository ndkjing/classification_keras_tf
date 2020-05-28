import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import tensorflow as tf
import numpy as np
print(tf.__version__)

model = tf.keras.models.load_model('mb2.h5')
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()


open('mb2_classify_f16_quant.tflite','wb').write(tflite_model)

# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.lite.Interpreter('mb2_classify_f16_quant.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input_details',input_details)
print('output_details',output_details)
# 使用随机数据作为输入测试 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# 函数 `get_tensor()` 会返回一份张量的拷贝。
# 使用 `tensor()` 获取指向张量的指针。
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)
# 使用随机数据作为输入测试 TensorFlow 模型。
# tf_results = model(tf.constant(input_data))
# print(tflite_results,tf_results)
# # 对比结果。
# for tf_result, tflite_result in zip(tf_results, tflite_results):
#     np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)