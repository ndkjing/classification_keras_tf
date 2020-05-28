"""
https://github.com/qubvel/efficientnet
"""
import keras_efficientnets as efn
import keras.backend as K
import keras.models as KM
import keras.layers as KL
import keras.optimizers as KO


def get_efficientnetb5(input_shape=(456,456,3),classes=9):
    model_b5 = efn.EfficientNetB5(weights=None)
    model_b5.load_weights('/home/create/jing/jing_vision/classification/weights/efficientnet/imagenet_weight/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment.h5')
    model_b5.layers.pop()
    model_b5.layers.pop()
    # head_model = KL.GlobalAveragePooling2D()(model_b5.layers[-1].output)
    head_model = KL.Dense(1024, activation='relu')(model_b5.layers[-1].output)
    head_model = KL.Dense(classes, activation='softmax')(head_model)
    model_b5_include_head = KM.Model(model_b5.input, head_model)
    sgd = KO.SGD(lr=0.001, momentum=0.9)
    model_b5_include_head.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_b5_include_head


def get_efficientnetb6(input_shape=(528,528,3),classes=9):
    model_b6 = efn.EfficientNetB6(weights=None)
    model_b6.load_weights(
        '/home/create/jing/jing_vision/classification/weights/efficientnet/imagenet_weight/efficientnet-b6_weights_tf_dim_ordering_tf_kernels_autoaugment.h5')
    model_b6.layers.pop()
    model_b6.layers.pop()
    # head_model = KL.GlobalAveragePooling2D()(model_b6.layers[-1].output)
    head_model = KL.Dense(1024, activation='relu')(model_b6.layers[-1].output)
    head_model = KL.Dense(classes, activation='softmax')(head_model)
    model_b6_include_head = KM.Model(model_b6.input, head_model)
    sgd = KO.SGD(lr=0.001, momentum=0.9)
    model_b6_include_head.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_b6_include_head


def get_efficientnetb7(input_shape=(600,600,3),classes=9):
    model_b7 = efn.EfficientNetB7(weights=None)
    model_b7.load_weights('/home/create/jing/jing_vision/classification/weights/efficientnet/imagenet_weight/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment.h5')
    model_b7.layers.pop()
    model_b7.layers.pop()
    # head_model = KL.GlobalAveragePooling2D()(model_b7.layers[-1].output)
    head_model = KL.Dense(1024, activation='relu')(model_b7.layers[-1].output)
    head_model = KL.Dense(classes, activation='softmax')(head_model)
    model_b7_include_head = KM.Model(model_b7.input, head_model)
    # sgd = KO.SGD(lr=0.001, momentum=0.9)
    # model_b7_include_head.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_b7_include_head