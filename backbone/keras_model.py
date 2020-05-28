import os, cv2, sys
from os import listdir
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from optimizer.rAdam import RectifiedAdam
import tensorflow as tf
print("----------------",tf.__version__,"is_gpu_available",tf.test.is_gpu_available())
from tensorflow.keras import backend

import tensorflow.keras.models as KM
import tensorflow.keras.layers as KL


# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from keras.applications.imagenet_utils import preprocess_input
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras import applications
# from sklearn.metrics import f1_score, fbeta_score


def f1(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    f1_score = backend.mean(2 * (p * r) / (p + r + backend.epsilon()))
    return f1_score


def get_vgg16(classes=9, input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.vgg16 import VGG16
    base_model = VGG16(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='elu', name='00', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='elu', name='01', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='11')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_vgg19(classes=9, input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.vgg19 import VGG19
    base_model = VGG19(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='01', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='11')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_resnetv2(classes=54, depth=50,input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2,ResNet101V2,ResNet152V2
    assert depth in [50, 101, 152]
    if depth == 50:
        base_model = ResNet50V2(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        # head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
        # head_model = KL.Dropout(0.5)(head_model)
        if classes == 2:
            head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
        else:
            head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model

    elif depth == 101:
        base_model = ResNet101V2(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        # head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
        # head_model = KL.Dropout(0.5)(head_model)
        if classes == 2:
            head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
        else:
            head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model

    else:
        base_model = ResNet152V2(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(1024, activation='relu', name='11', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        if classes == 2:
            head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
        else:
            head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model


def get_resnet(classes=54, depth=50,input_shape=(224, 224, 3),base_layer_trainable=False):
    assert depth in [50,101,152]
    from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
    if depth ==50:
        base_model = ResNet50(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        # head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
        # head_model = KL.Dropout(0.5)(head_model)
        if classes == 2:
            head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
        else:
            head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model
    elif depth==101:
        base_model = ResNet101(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        # head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
        # head_model = KL.Dropout(0.5)(head_model)
        if classes == 2:
            head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
        else:
            head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model
    else:
        base_model = ResNet152(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(1024, activation='relu', name='11', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        if classes == 2:
            head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
        else:
            head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model


def get_xception(classes=54, input_shape=(299, 299, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.xception import Xception
    base_model = Xception(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalAveragePooling2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.2)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.2)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_densenet(classes=9, depth =121,input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
    assert depth in [121, 169, 201]
    if depth==121:
        base_model = DenseNet121(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable =base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model

    elif depth==169:
        base_model = DenseNet169(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(1024, activation='relu', name='01', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(classes, activation='softmax', name='11')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model

    else:
        base_model = DenseNet201(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = base_layer_trainable
        head_model = KL.GlobalMaxPool2D()(base_model.output)
        head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
        head_model = KL.Dropout(0.5)(head_model)
        head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
        model = KM.Model(inputs=base_model.input, outputs=head_model)
        return model


def get_inceptionresnetv2(classes=9, input_shape=(229, 229, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='01', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='11')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_inceptionv3(classes=9, input_shape=(229, 229, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='00', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='01', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='11')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_mobilenet(classes=2, input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.mobilenet import MobileNet
    base_model = MobileNet(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)

    return model


def get_mobilenetv2(classes=9, input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    base_model = MobileNetV2(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    # head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
    # head_model = KL.Dropout(0.5)(head_model)
    if classes == 2:
        head_model = KL.Dense(classes, activation='sigmoid', name='3333')(head_model)
    else:
        head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_nasnetlarge(classes=9, input_shape=(331, 331, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.nasnet import NASNetLarge
    base_model = NASNetLarge(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model


def get_nasnetmobile(classes=9, input_shape=(224, 224, 3),base_layer_trainable=False):
    from tensorflow.keras.applications.nasnet import NASNetMobile
    base_model = NASNetMobile(include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = base_layer_trainable
    head_model = KL.GlobalMaxPool2D()(base_model.output)
    head_model = KL.Dense(1024, activation='relu', name='0000', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(1024, activation='relu', name='1111', kernel_initializer='he_uniform')(head_model)
    head_model = KL.Dropout(0.5)(head_model)
    head_model = KL.Dense(classes, activation='softmax', name='3333')(head_model)
    model = KM.Model(inputs=base_model.input, outputs=head_model)
    return model



if __name__ == '__main__':
    # 返回未compile模型
    from tensorflow.keras.optimizers import Adam, SGD
    model = get_vgg16()
    model.summary()
    # radam = RectifiedAdam(lr=1e-4)
    adam = Adam(lr=1e-5)
    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[f1])
    print(model.input)
