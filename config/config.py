# 模型及其输入尺寸

model_input_shape = {'vgg16': 224,
                     'vgg19': 224,
                     'resnet': 224,
                     'resnet_v2': 224,
                     'inception_resnet_v2': 229,
                     'inception_v3': 229,
                     'xception': 299,
                     'mobilenet': 224,
                     'mobilenet_v2': 224,
                     'densenet': 224,
                     'nasnet_large': 331,
                     'nasnet_mobile': 224,
                     'resnext': 224,
                     'seresnet': 224,
                     'efficientb7': 600,
                     'seresnext': 224,
                     'seinceptionresnetv2': 299,
                     'efficientnetb5': 456,
                     'efficientnetb6': 528,
                     'efficientnetb7': 600,
                     }

batch_size = 64
epochs=500

# 训练数据文件夹  文件夹中包含trian val   train/val 中类型按文件名划分
datapath = '/Data/jing/black_river_classify_data'
#保存训练过程中权重路径
save_weight_path = '/Data/jing/weights/classification/keras_tf'
# 预训练模型保存路径
pretrain_weight_path = '/Data/jing/weights/classification/keras_tf/pretrain'