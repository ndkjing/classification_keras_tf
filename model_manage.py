import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification.keras_model.keras_model import *
from classification.keras_model.resnext import get_resnext
from classification.keras_model.efficientnet.efficientnet import get_efficientnetb5,get_efficientnetb6,get_efficientnetb7


def get_all_model_by_name(model_name='mobilenetv2'):
    assert model_name in model_keras.keys() or model_name in model_self.keys()
    function_name = 'get_' + model_name
    model = globals()[function_name]()
    adam = Adam(lr=1e-2)
    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = get_all_model_by_name(model_name='efficientnetb6')
    # print(weights.name)
    model.summary()