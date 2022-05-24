import keras.regularizers
import pandas as pd
import csv as csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Lasso, Ridge
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D,MaxPool1D,Embedding,LSTM,GRU
import keras.backend as K
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.regularizers import l2
from collections import Counter
from keras.callbacks import LearningRateScheduler
import math

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return initial_lrate

def lasso_regression():
    lasso_model = Lasso()
    return lasso_model

def ridge_regression():
    ridge_model = Ridge()
    return ridge_model


# 神经网络模型 三个参数为名称
def CNN_GRU(train, test, name):

    # 读取数据
    data_train = pd.read_csv(
        '/home/ubuntu/SDP/datasets/mapped data' + train + '.csv')

    data_test = pd.read_csv(
        '/home/ubuntu/SDP/datasets/mapped data' + test + '.csv')


    # 训练数据去除不用的列
    data_train = data_train.drop(['filename'], axis=1)

    # 分离训练要用的X，与y
    X_train = data_train.drop('bugs', axis=1)
    y_train = pd.DataFrame(data_train['bugs'], columns=['bugs'])

    # 测试数据处理同上
    data_test = data_test.drop(['filename'], axis=1)

    X_test = data_test.drop('bugs', axis=1)
    y_test = pd.DataFrame(data_test['bugs'], columns=['bugs'])

    # 定义一些变量
    length = X_train.shape[1]
    cols_test = X_test.shape[0]
    input_shape = (X_train.shape[1], 1)


    # 归一化，提高模型计算性能
    scaler = MinMaxScaler()
    MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    # 重复实验，range内设置重复实验次数
    for i in range(1):
        # CNN模型构建
        model = Sequential()
        model.add(Embedding(output_dim=32, input_dim=X_train.shape[0], input_length=length))

        model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPool1D(pool_size=2))
        model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(MaxPool1D(pool_size=2))
        model.add(GRU(10, dropout=0.3, activation='relu', return_sequences=True))
        # model.add(GRU(10, dropout=0.3, activation='relu'))


        model.add(Flatten())
        model.add(Dense(160, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
        # model.add(Dropout(0.5))
        # model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
        # model.add(Dropout(0.5))

        model.add(Dense(1))


        # 设置优化器的一些参数，可以调整
        Adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # 编译模型
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        model.summary()

        # 设置早停，monitor为检测的指标，patience为周期
        early_stop = EarlyStopping(monitor='mae', patience=5)

        # 设置学习率衰减
        # reduce_lr = LearningRateScheduler()

        # 模型训练，参数可调整
        history = model.fit(X_train, y_train, epochs=50, batch_size=128, shuffle=True, validation_split=0.2)

        res = model.predict(X_test)

        predictions_y_round = np.rint(res)


        # 选择写入结果的路径
        f = open(r'/home/ubuntu/SDP/result/Ranking/crossVersion/CNN_GRU_2/' + name + '_' + str(i) + '.csv', 'a+', encoding='utf-8', newline='')

        f.writelines('file_index,original_value,predict_value\n')

        flag = 0

        for item_origin, item_predict in zip(y_test.values.flatten(), predictions_y_round.flatten()):
            print(item_origin)
            print(item_predict)
            f.writelines(str(flag) + ',' + str(item_origin) + ',' + str(item_predict) + '\n')
            flag += 1

        f.close()

def baseline(train, test, name):

    data_train = pd.read_csv(
        '/home/ubuntu/SDP/datasets/mapped data' + train + '.csv')

    data_test = pd.read_csv(
        '/home/ubuntu/SDP/datasets/mapped data' + test + '.csv')

    data_train = data_train.drop(['filename'], axis=1)

    X_train = data_train.drop('bugs', axis=1)
    y_train = pd.DataFrame(data_train['bugs'], columns=['bugs'])

    data_test = data_test.drop(['filename'], axis=1)

    X_test = data_test.drop('bugs', axis=1)
    y_test = pd.DataFrame(data_test['bugs'], columns=['bugs'])

    length = X_train.shape[1]
    cols_test = X_test.shape[0]
    input_shape = (X_train.shape[1], 1)

    # 归一化
    scaler = MinMaxScaler()
    MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    for i in range(1):
        # 模型构建，可选择上面定义的一些模型，模型参数可调整
        model = ridge_regression()

        model.fit(X_train, y_train.values.flatten())

        res = model.predict(X_test)

        predictions_y_round = np.rint(res)

        # 储存结果路径
        f = open(r'/home/ubuntu/SDP/result/Ranking/crossVersion/ridge/' + name + '.csv', 'a+', encoding='utf-8',
                 newline='')

        f.writelines('file_index,original_value,predict_value\n')

        flag = 0

        for item_origin, item_predict in zip(y_test.values.flatten(), predictions_y_round.flatten()):
            print(item_origin)
            print(item_predict)
            f.writelines(str(flag) + ',' + str(item_origin) + ',' + str(item_predict) + '\n')
            flag += 1

        f.close()

if __name__ == '__main__':
    #用于做cross version的文件名称，前一个用于训练，后一个用于预测
    file_list = [[r'/camel-1.4', r'/camel-1.6', 'camel'],[r'/jedit-4.0', r'/jedit-4.1', 'jedit'],[r'/lucene-2.0', r'/lucene-2.2', 'lucene'], [r'/xalan-2.5', r'/xalan-2.6', 'xalan'], [r'/xerces-1.2', r'/xerces-1.3', 'xerces'], [r'/synapse-1.1', r'/synapse-1.2', 'synapse'], [r'/poi-2.5', r'/poi-3.0', 'poi']]
    # file_list = [[r'/synapse-1.1', r'/synapse-1.2', 'synapse']]

    for item in file_list:
        res = CNN_GRU(item[0], item[1], item[2])



