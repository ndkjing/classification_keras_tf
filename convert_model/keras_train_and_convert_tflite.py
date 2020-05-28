import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import numpy as np
import tensorflow as tf
print(tf.__version__,tf.test.is_gpu_available())
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMG_SHAPE = (224, 224, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable =True
head_model = tf.keras.layers.GlobalMaxPool2D()(base_model.output)
head_model =tf.keras.layers.Dense(1024, activation='relu', name='00')(head_model)
head_model = tf.keras.layers.Dropout(0.5)(head_model)
# head_model = tf.keras.layers.Dense(1024, activation='relu', name='11', kernel_initializer='he_uniform')(head_model)
# head_model =tf.keras.layers.Dropout(0.5)(head_model)
head_model = tf.keras.layers.Dense(10, activation='softmax', name='10')(head_model)
model = tf.keras.Model(inputs=base_model.input, outputs=head_model)
# feature_batch = base_model(np.random.random([16,224,224,3]))
# print(feature_batch.shape)
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
# prediction_layer = tf.keras.layers.Dense(10)
#
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)
# model = tf.keras.Sequential([
#   base_model,
#   global_average_layer,
#   prediction_layer
# ])

base_learning_rate = 3e-4
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(horizontal_flip=True,rescale = 1.0 / 255.0)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_data_gen = train_datagen.flow_from_directory(batch_size=32,
                                                   directory='/home/create/jing/jing_vision/detection/pth/ssd.pytorch/hand_gasture/color_hand_gasture/train/',
                                                   shuffle=True,
                                                   target_size=(224, 224))
val_data_gen = test_datagen.flow_from_directory(batch_size=32,
                                                   directory='/home/create/jing/jing_vision/detection/pth/ssd.pytorch/hand_gasture/color_hand_gasture/val/',
                                                   shuffle=True,
                                                   target_size=(224, 224))
print(val_data_gen.class_indices)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_data_gen),
    epochs=50,
    validation_data=val_data_gen,
    validation_steps=len(val_data_gen),
    verbose=1
)

model.save('mb2.h5')

convert = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = convert.convert()


open('mb2_classify.tflite','wb').write(tflite_model)

# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.lite.Interpreter('mb2_classify.tflite')
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

# 使用随机数据作为输入测试 TensorFlow 模型。
tf_results = model(tf.constant(input_data))
print(tflite_results,tf_results)
# 对比结果。
for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)