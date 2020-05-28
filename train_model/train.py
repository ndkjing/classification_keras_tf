import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))



from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD

from keras_tf.dataset.data_generate import load_dataset_to_memory,load_generate_data_from_file
from keras_tf.backbone.keras_model import *
from keras_tf.config import config
# 设置显存占用按需求增长
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)
# K.clear_session()  # 清楚设置GPU回话以免混淆后面的训练session



model_name_map = {
    'vgg16': get_vgg16,
    'vgg19': get_vgg19,
    'resnet': get_resnet,
    'resnet_v2': get_resnetv2,
    'inception_resnet_v2': get_inceptionresnetv2,
    'inception_v3': get_inceptionv3,
    'xception': get_xception,
    'mobilenet': get_mobilenet,
    'mobilenet_v2': get_mobilenetv2,
    'densenet': get_densenet,
    'nasnet_large': get_nasnetlarge,
    'nasnet_mobile': get_nasnetmobile,
    'resnext': None,
    'seresnet': None,
    'efficientb7': None

}

# 绘制学习曲线
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['accuracy'], color='orange', label='test')
    # save plot to file
    plt.savefig('acc_loss.png')
    plt.close()


# 绘制Kfold每次的score
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.savefig('scores.png')


# 绘制Kfold每次acc与loss
def summarize_diagnostics_kfold(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(
            histories[i].history['val_loss'],
            color='orange',
            label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(
            histories[i].history['val_accuracy'],
            color='orange',
            label='test')
    plt.savefig('kfold_acc_loss.png')


# 在图片类别按文件夹分类的图像中绘制混淆矩阵
def plot_sonfusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')


def draw_confusion_matrix_from_file(model=None, target_size=(
        224, 224, 3), test_dir='/Data/jing/weather_classify/test/'):
    assert model is not None
    test = os.listdir(test_dir)
    num_classes = len(test)
    print(test)
    imgs, ys = list(), list()

    for filename_index in range(1, num_classes + 1):
        filename = os.path.join(test_dir, str(filename_index)) + '/'
        print(filename)
        for image_name in os.listdir(filename):
            # print(image_name)
            image_full_path = os.path.join(filename, image_name)
            # print(image_full_path)
            img = tf.keras.preprocessing.image.load_img(image_full_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img.astype('float32') / 255
            y = filename_index
            imgs.append(img)
            ys.append(y)
    X = np.asarray(imgs, dtype='float32')
    y = np.asarray(ys, dtype='uint8')
    print(X.shape)
    print(y.shape)
    # one-hot结果转为数字标签
    y_pre_temp = model.predict(X)
    y_pre = []
    for row in range(y_pre_temp.shape[0]):
        #         print(y_pre[row,:].argmax())
        y_pre.append(y_pre_temp[row, :].argmax() + 1)
    y_pre = np.asarray(y_pre)
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(y, y_pre)
    print(confusion_mat)
    plot_sonfusion_matrix(confusion_mat, classes=range(1, num_classes + 1))


def run(model_name):
    # 构建模型
    model = model_name_map[model_name](classes=len(os.listdir(os.path.join(config.datapath,'train'))))
    adam = Adam(lr=1e-3)
    sgd = SGD(lr=1e-3, momentum=0.9)
    model.compile(
        optimizer=sgd,
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy'])
    # 创建模型保存路径
    model_weights_path = os.path.join(
        os.path.abspath('__file__'), 'weights')
    print('模型保存路劲：', model_weights_path)
    if not os.path.exists(config.save_weight_path):
        os.mkdir(model_weights_path)

    # 回调保存模型
    model_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_weights_path, "%s{epoch:03d}-{val_loss:.2f}.h5"%(model_name)),
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 period=2)
    try:
        print('加载模型权重')
        model.load_weights(
            os.path.join(config.pretrain_weight_path,model_name+'.h5'),
            by_name=True)
    except:
        print('模型不存在或尺寸不匹配,跳过加载权重')
    model.summary()
    # 加载数据

    ########## 两种加载数据方式
    # 一、直接加载全部数据到内存中
    load_alldata_to_memory = False  # 使用方式一 使用方式一可以使用generator.fit使用按数据集特征normal 使用Kfold
    use_kfold = False
    use_data_standardization = False
    if load_alldata_to_memory:
        trainX, trainY, testX, testY = load_dataset_to_memory(
            num_classes=54, input_shape=(
                config.model_input_shape[model_name], config.model_input_shape[model_name], 3), )
        print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
        print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
        if use_data_standardization:  # 使用标准化图像  数据量少的话（<50000）可能均值会偏离0
            print('Statistics train=%.3f (%.3f), test=%.3f (%.3f)' %
                  (trainX.mean(), trainX.std(), testX.mean(), testX.std()))
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True)
            # calculate mean on training dataset
            datagen.fit(trainX)
            print('datagen.mean:', datagen.mean)
            # prepare an iterators to scale images
            train_iterator = datagen.flow(trainX, trainY, batch_size=32)
            test_iterator = datagen.flow(testX, testY, batch_size=32)
            iterator = datagen.flow(
                trainX,
                trainY,
                batch_size=trainX.shape[0],
                shuffle=False)
            # get a batch
            batchX, batchy = iterator.next()
            # pixel stats in the batch
            print(
                'stand:',
                trainX.shape[0],
                batchX.shape,
                batchX.mean(),
                batchX.std())
            print('Batches train=%d, test=%d' %
                  (len(train_iterator), len(test_iterator)))
            model.fit_generator(
                train_iterator,
                validation_data=test_iterator,
                callbacks=[model_callback],
                steps_per_epoch=len(train_iterator),
                epochs=500,
                verbose=1)
            # evaluate keras_tf
        if use_kfold:  # 是否使用Kfold
            scores, histories = list(), list()
            # prepare cross validation
            kfold = KFold(5, shuffle=True, random_state=1)
            # enumerate splits
            for train_ix, test_ix in kfold.split(trainX):
                trainX_, trainY_, valX_, valY_ = trainX[train_ix], trainY[
                    train_ix], trainX[test_ix], trainY[test_ix]
                history = model.fit(
                    trainX_,
                    trainY_,
                    epochs=1,
                    batch_size=32,
                    validation_data=(
                        valX_,
                        valY_),
                    verbose=1)
                # evaluate keras_tf
                _, acc = model.evaluate(valX_, valY_, verbose=1)
                print('> %.3f' % (acc * 100.0))
                scores.append(acc)
                histories.append(history)
            summarize_diagnostics_kfold(histories)
            # summarize estimated performance
            summarize_performance(scores)

        history = model.fit(
            trainX,
            trainY,
            epochs=2,
            batch_size=config.batch_size,
            validation_data=(
                testX,
                testY),
            verbose=1)
        # evaluate keras_tf
        _, acc = model.evaluate(testX, testY, verbose=1)
        summarize_diagnostics(history)
    # 二、每次读取一个batchsize数据到内存  不适用clr
    else:
        train_it, test_it = load_generate_data_from_file(input_shape=(config.model_input_shape[model_name],
                                                                      config.model_input_shape[model_name],
                                                                      3),
                                                         batch_size=config.batch_size,
                                                         )
        # 显示均值与方差
        # batchX, batchy = train_it.next()
        # print('stand:', batchX.shape, batchX.mean(), batchX.std())
        lr_find = False  # 使用学习率查找
        clr_flag = False  # 使用循环学习率
        if not clr_flag:
            history = model.fit_generator(train_it,
                                          steps_per_epoch=len(train_it),
                                          callbacks=[model_callback],
                                          validation_data=test_it,
                                          validation_steps=len(test_it),
                                          epochs=config.epochs,
                                          verbose=1
                                          )
        ### 使用学习率查找
        """
    
        if lr_find:
            from utils.optimizer.learningratefind import LearningRateFinder
            lrf = LearningRateFinder(keras_tf)
            lrf.find(
                train_it,
                1e-10, 1e+1,
                stepsPerEpoch=len(train_it),
                batchSize=200)

            # resulting plot to disk
            lrf.plot_loss()
            plt.savefig('lrf.jpg')
        # keras_tf.summary()
        # 使用循环学习率
        if clr_flag:
            from utils.optimizer.clr_callback import CyclicLR
            clr = CyclicLR(
                base_lr=1e-3,
                max_lr=1e-1,
                # step_size=30,
            )

            history = keras_tf.fit_generator(train_it,
                                          steps_per_epoch=len(train_it),
                                          callbacks=[best_model,clr],
                                          validation_data=test_it,
                                          validation_steps=len(test_it),
                                          epochs=epochs,
                                          verbose=1)
            _, acc = keras_tf.evaluate_generator(test_it, steps=len(test_it), verbose=1)
            print('> %.3f' % (acc * 100.0))
        """
        summarize_diagnostics(history)
        draw_confusion_matrix_from_file(model=model, target_size=(config.model_input_shape[model_name],
                                                                  config.model_input_shape[model_name],
                                                                  3))


# TODO 删除错误图像
def delete_error_image():
    pass


if __name__ == '__main__':
    print(tf.__version__)
    ###   赋值训练模型名称
    model_name ='mobilenet'
    run(model_name=model_name)
