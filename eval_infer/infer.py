import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 设置使用GPU序号

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
# 查看显卡状态
K.tensorflow_backend._get_available_gpus()


from keras.models import load_model
model = load_model('/home/create/jing/jing_vision/implement/model_sgd_mobilenet.h5') # 加载模型
model.summary()


# 方式一 使用 ImageDatagenarate 从验证集中读取


train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rescale = 1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
batch_size = 32
target_size= 224
# weather_base_dir = "/Data/jing/huawei/train_data/"
base_dir = "/Data/jing/black_river_classify_data/classify_day/"
train_it = train_datagen.flow_from_directory(directory=os.path.join(base_dir,'train'),
                                             target_size=(target_size, target_size),
                                             batch_size=batch_size,
                                             )
test_it = test_datagen.flow_from_directory(directory=os.path.join(base_dir,'val'),
                                             target_size=(target_size, target_size),
                                           batch_size=batch_size,
                                           )

testX,testy = test_it.next()


# 验证测试机准确率
# _,acc = model.evaluate_generator(test_it)
# print('测试集准确率为：',acc)

# 使用单独加载指定文件夹数据的方式数据

batch_size = 32
target_size= 224
test_img_dir = '/Data/jing/black_river_classify_data/river_hard_sample'
testX = []
testy=[]  # 若测试图片有标签可以添加标签
for i,img_name in enumerate(os.listdir(test_img_dir)):
    if i<batch_size:
        img = load_img(os.path.join(test_img_dir,img_name),target_size=(target_size,target_size))
        img = img_to_array(img)/255.0
        testy.append([1,0])
        testX.append(img)
testX = np.asarray(testX)
testy = np.asarray(testy)
print(testX.shape,testy.shape)

# 设置可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

pre = model.predict(testX)

map_dict = {0: 'black', 1: 'no_black'}
plt.figure(figsize=(100, 100))
for i in range(batch_size):
    ax = plt.subplot(math.ceil(32 ** (0.5)), math.ceil(32 ** (0.5)), i + 1)
    plt.axis('off')
    plt.imshow(testX[i, :, :, :])
    title = u'预测：' + map_dict[np.argmax(pre[i, :])] + u'  标签：' + map_dict[np.argmax(testy[i, :])]
    color = 'black' if np.argmax(pre[i, :]) == np.argmax(testy[i, :]) else 'red'
    plt.title(title, fontsize=50, color=color)

plt.show()
