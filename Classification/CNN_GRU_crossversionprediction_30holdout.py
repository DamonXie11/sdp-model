import keras.regularizers
import pandas as pd
import csv as csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D,MaxPool1D,Embedding,LSTM,GRU
from keras.metrics import TrueNegatives,TruePositives,FalseNegatives,FalsePositives
import keras.backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.regularizers import l2
from collections import Counter
from keras.callbacks import LearningRateScheduler
from sklearn.naive_bayes import GaussianNB
import math

# 训练过程中，观察F1值
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

# 训练过程中，观察G值
def metric_G_measure(y_true,y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    TNR = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    G_measure = 2 * recall * TNR / (recall + TNR)

    return G_measure

# 学习率衰减
def step_decay(epoch):
    initial_lrate = 0.001
    # drop = 0.5
    # epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return initial_lrate

# 输入两个数组，计算评价指标，返回acc, recall, precision, F1score, G_measure
def cal_metric_measure(origin, predict):
    length = len(origin)
    len1 = len(predict)
    values = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'F1': 0}
    if length != len1:
        print("cal_metric_F_measure 输入长度不一致")
        return
    TP, FP, FN, TN = 0, 0, 0, 0
    for item_origin, item_predict in zip(origin, predict):
        #         print(item_origin)
        #         print(item_predict)
        if item_origin == item_predict == item_predict == 1:
            TP += 1
        elif item_origin == 1 and item_predict == 0:
            FN += 1
        elif item_origin == 0 and item_predict == 1:
            FP += 1
        elif item_origin == item_predict == item_predict == 0:
            TN += 1

    acc = (TP + TN) / length
    if TN + TP == 0:
        TNR = 0
    else:
        TNR = TN / (TN + TP)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        F1score = 0
    else:
        F1score = 2 * precision * recall / (precision + recall)
    if recall + TNR == 0:
        G_measure = 0
    else:
        G_measure = 2 * recall * TNR / (recall + TNR)

    return [acc, recall, precision, F1score, G_measure]

# 定义CNN_GRU模型
def CNN_GRU(train, test, name):

    # 读取训练和测试训练数据
    data_train = pd.read_csv(
        '/home/ubuntu/SDP/datasets/mapped data' + train + '.csv')

    data_test = pd.read_csv(
        '/home/ubuntu/SDP/datasets/mapped data' + test + '.csv')

    # 将标签改为一致，将有有缺陷的文件个数即bug个数大于1的个数改为1，即有缺陷label数量为1，没有缺陷的label数量改为0
    for i in range(len(data_train['bugs'])):
        if data_train['bugs'][i] > 1:
            data_train['bugs'][i] = 1


    data_train = data_train.drop(['filename'], axis=1)

    X_train = data_train.drop('bugs', axis=1)
    y_train = pd.DataFrame(data_train['bugs'], columns=['bugs'])

    # 同上处理测试数据集
    for i in range(len(data_test['bugs'])):
        if data_test['bugs'][i] > 1:
            data_test['bugs'][i] = 1

    data_test = data_test.drop(['filename'], axis=1)

    X_test = data_test.drop('bugs', axis=1)
    y_test = pd.DataFrame(data_test['bugs'], columns=['bugs'])

    # In[4]:
    # 仅在训练集中使用随机过采样，将有缺陷和没有缺陷的文件数量的比例调整为1：1，仅仅对训练数据进行随机过过采样
    ros = RandomOverSampler(random_state=0)
    X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)


    # 定义一些变量
    length = X_train.shape[1]
    cols_test = X_test.shape[0]
    input_shape = (X_train.shape[1], 1)


    # 归一化，范围0～1
    MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train = MinMaxScaler().fit_transform(X_train_overSampling)
    X_test = MinMaxScaler().fit_transform(X_test)

    y_train = y_train_overSampling.values.reshape(y_train_overSampling.shape[0], 1)
    y_train = MinMaxScaler().fit_transform(y_train)


    # 重复实验
    for i in range(30):
        # CNN模型构建
        model = Sequential()
        model.add(Embedding(output_dim=30, input_dim=X_train.shape[0], input_length=length,
                            embeddings_regularizer=keras.regularizers.l2(0.1)))

        model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=input_shape))
        # model.add(MaxPool1D(pool_size=2))
        # model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=input_shape))
        # model.add(MaxPool1D(pool_size=2))
        # model.add(LSTM(20, dropout=0.3, activation='relu', return_sequences=True))
        model.add(GRU(10, dropout=0.3, activation='relu', return_sequences=True))


        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid', kernel_initializer="glorot_normal"))


        # 编译模型
        model.compile(optimizer='Adam', loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), metric_F1score,
                               metric_G_measure])

        model.summary()

        # 学习率衰减设置，需要在model.fit的callback属性中设置
        reduce_lr = LearningRateScheduler(step_decay)

        history = model.fit(X_train, y_train, epochs=50, batch_size=64, shuffle=True, validation_split=0.2, callbacks=[reduce_lr])

        # 利用evaluate函数计算评价指标
        score = model.evaluate(X_test, y_test.values, verbose=1)
        print(score)

        # score函数计算的评价指标储存路径
        f = open(
            r'/home/ubuntu/SDP/result/CNN+GRU/CrossVersion/3_26/3_26_evaluate/' + name + '_result.csv',
            'a', encoding='utf-8', newline='' "")

        csv_writer = csv.writer(f)
        csv_writer.writerow(score)
        f.close()

        # 预测
        y_predict = model.predict(X_test)
        y_predict = np.rint(y_predict)
        y_test = np.rint(y_test)

        # 使用cal_metric_measure计算评价指标
        res = cal_metric_measure(y_test.values.reshape(cols_test, ), y_predict.reshape(cols_test, ))
        print('res', res)

        if res != 0:
            # 计算手工计算结果计算储存路径
            f1 = open(
                r'/home/ubuntu/SDP/result/CNN+GRU/CrossVersion/3_26/3_26_calculate/' + name + '_result.csv',
                'a', encoding='utf-8', newline='' "")

            csv_writer1 = csv.writer(f1)

            # csv_writer.writerow(["Y","Y_predict"])

            csv_writer1.writerow(res)
            f1.close()


if __name__ == '__main__':
    file_list = [[r'/camel-1.4', r'/camel-1.6', 'camel'],[r'/jedit-4.0', r'/jedit-4.1', 'jedit'],[r'/lucene-2.0', r'/lucene-2.2', 'lucene'], [r'/xalan-2.5', r'/xalan-2.6', 'xalan'], [r'/xerces-1.2', r'/xerces-1.3', 'xerces'], [r'/synapse-1.1', r'/synapse-1.2', 'synapse'], [r'/poi-2.5', r'/poi-3.0', 'poi']]
    # file_list = [[r'/synapse-1.1', r'/synapse-1.2', 'synapse']]

    for item in file_list:
        res = CNN_GRU(item[0], item[1], item[2])



