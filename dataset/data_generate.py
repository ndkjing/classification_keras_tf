import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# from tensorflow.keras.preprocessing import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
# 设置图片读取允许对任何图像格式
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras_tf.config import config

def one_hot_encode(str_y, num_classes):
    encoding = np.zeros(num_classes, dtype='uint8')
    index = int(dic[str_y])
    encoding[index] = 1
    return encoding


# 直接加载图片
def load_weather_dataset(img_path=config.datapath, num_classes=9, input_shape=(456,456,3)):
    imgs, ys = list(), list()
    data = pd.read_csv(config.datapath)
    filename_list = list(data['FileName'])
    y_list = list(data['type'])
    for filename_index in range(len(filename_list)):
        img = load_img(img_path + filename_list[filename_index], target_size=input_shape)
        img = img_to_array(img)
        img = img.astype('float32') / 255
        y = one_hot_encode(y_list[filename_index], num_classes)
        imgs.append(img)
        ys.append(y)
    X = np.asarray(imgs, dtype='float32')
    y = np.asarray(ys, dtype='uint8')
    return X, y

dic=None
# 将按目录分好类的train  val加载到内存中并存储位npz文件方便下次读取
# 参数：文件路径存放train val的文件夹  类别数量  输入图片形状
def load_dataset_to_memory(num_classes=40,input_shape=(456,456,3)):
    global dic
    save_npz_name=config.datapath+'save.npz'
    if not os.path.exists(save_npz_name):
        print('数据 loading..')
        for subpath in ['train','val']:
            print(subpath)
            imgs, ys = list(), list()
            abs_sub_path = os.path.join(config.datapath,subpath)
            if dic is None:
                categorical = os.listdir(abs_sub_path)
                values = np.array(categorical)
                label_encoder = LabelEncoder()
                integer_encoded = label_encoder.fit_transform(values)
                dic = {categorical[i]:integer_encoded[i] for i in range(len(categorical))}
                print(dic)
            for categorical_file in os.listdir(abs_sub_path):
                abs_categorical_file  = os.path.join(abs_sub_path,categorical_file)
                for image_name in os.listdir(abs_categorical_file):
                    abs_image_name = os.path.join(abs_categorical_file,image_name)
                    img = load_img(abs_image_name, target_size=input_shape)
                    img = img_to_array(img)
                    img = img.astype('float32')
                    # y = one_hot_encode(categorical_file, num_classes)
                    encoding = np.zeros(num_classes, dtype='uint8')
                    index = int(dic[categorical_file])
                    encoding[index] = 1
                    y =  encoding
                    imgs.append(img)
                    ys.append(y)
            if subpath=='train':
                trainX = np.asarray(imgs, dtype='float32')
                trainy = np.asarray(ys, dtype='uint8')
            elif subpath=='val':
                testX = np.asarray(imgs, dtype='float32')
                testy = np.asarray(ys, dtype='uint8')
        np.savez_compressed(save_npz_name,trainX=trainX,trainy=trainy,testX=testX,testy=testy)
    npz_data = np.load(save_npz_name)
    return npz_data['trainX'],npz_data['trainy'],npz_data['testX'],npz_data['testy']


def load_generate_data_from_file(input_shape=(456,456,3), batch_size=32):
    train_datagen = ImageDataGenerator(
        # rotation_range=15,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.01,
        # zoom_range=[0.9, 1.25],
        horizontal_flip=True,
        # vertical_flip=False,
        # brightness_range=[0.5, 1.5],
        # featurewise_center = True,
        # featurewise_std_normalization = True,
        rescale = 1.0 / 255.0,
        # rotation_range=30,
        # zoom_range=0.15,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.15,
        # horizontal_flip=True,
        # fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1.0 / 255.0
    )
    # 使用ImageNet上均值做图像中心化
    # train_datagen.mean = np.asarray([0.58112579, 0.52210326, 0.46159332])*255  #[0.485, 0.456, 0.406]
    # train_datagen.std = np.asarray([0.23822173, 0.25246721, 0.26863971])*255 # [0.229, 0.224, 0.225]
    # test_datagen.mean = np.asarray([0.58112579, 0.52210326, 0.46159332])*255  #[0.485, 0.456, 0.406]
    # test_datagen.std = np.asarray([0.23822173, 0.25246721, 0.26863971])*255  # [0.229, 0.224, 0.225]

    train_it = train_datagen.flow_from_directory(directory=os.path.join(config.datapath,'train'),
                                                 target_size=(input_shape[0], input_shape[1]),
                                                 batch_size=batch_size,
                                                 )
    test_it = test_datagen.flow_from_directory(directory=os.path.join(config.datapath,'val'),
                                                 target_size=(input_shape[0], input_shape[1]),
                                               batch_size=batch_size,
                                               )

    return train_it, test_it




