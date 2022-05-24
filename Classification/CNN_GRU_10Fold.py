import keras.regularizers
import pandas as pd
import csv as csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import time
import math
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D,MaxPool1D,Embedding,GRU,LSTM,SimpleRNN
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

def cal_metric_measure(origin, predict):
    length = len(origin)
    len1 = len(predict)
    values = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'F1': 0}
    if (length != len1):
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


    if (TN + TP == 0 or TP + FP == 0 or TP + FN == 0):
        return 0
    acc = (TP + TN) / length
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    TNR = TN / (TN + TP)

    F1score = 2 * precision * recall / (precision + recall)

    G_measure = 2 * recall * TNR / (recall + TNR)

    return [acc, recall, precision, F1score, G_measure]

def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    TNR = TN/(TN+FP)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)

    return F1score

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

def CNN_GRU_model(input_dim, input_length):
    # CNN+GRU模型构建
    model = Sequential()
    model.add(Embedding(output_dim=30, input_dim=input_dim, input_length=input_length,
                        embeddings_regularizer=keras.regularizers.l2(0.1)))
    model.add(Conv1D(10, kernel_size=5, activation='relu', kernel_initializer="he_normal"))
    model.add(MaxPool1D(pool_size=2))

    model.add(GRU(10, dropout=0.3, activation='relu', return_sequences=True))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.5))

    return model

def step_decay(epoch):
    initial_lrate = 0.001
    # drop = 0.5
    # epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    # initial_learning_rate = 0.001,
    # decay_steps = batch_size * 1000,
    # decay_rate = 1,
    # staircase = False
    return initial_lrate

def CNN_model(item):

    # 读取训练数据
    data_all = pd.read_csv(
        r'/home/ubuntu/SDP/datasets/mapped data' + item + '.csv')

    # 将分类的标签转换，有缺陷的为1，没有缺陷的为0
    for i in range(len(data_all['bugs'])):
        if data_all['bugs'][i] > 1:
            data_all['bugs'][i] = 1

    data = data_all.drop(['filename'], axis=1)

    X_data = data.drop('bugs', axis=1)
    y_data = pd.DataFrame(data['bugs'], columns=['bugs'])

    num_epochs = 50

    # 使用分层的KFold，即按照原本y/分类标签的比例划分
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(X_data.values, y_data.values):
        # print('train_index', train_index)
        # print('X_train', X_train.shape)
        X_train, X_test = X_data.values[train_index], X_data.values[test_index]
        y_train, y_test = y_data.values[train_index], y_data.values[test_index]

        # 仅在训练集中使用随机过采样
        ros = RandomOverSampler(random_state=0)
        X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)

        # 定义一些变量
        length = X_train.shape[1]
        input_shape = (X_train.shape[1], 1)
        cols_test = X_test.shape[0]
        input_dim = X_train.shape[0]

        # 归一化
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train_overSampling)
        X_test = MinMaxScaler().fit_transform(X_test)
        y_train = y_train_overSampling.reshape(y_train_overSampling.shape[0], 1)
        y_train = MinMaxScaler().fit_transform(y_train)

        # 定义模型
        model = CNN_GRU_model(input_dim, length)


        # 编译模型
        model.compile(optimizer='Adam', loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), metric_F1score, metric_G_measure])

        model.summary()

        reduce_lr = LearningRateScheduler(step_decay)

        history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, shuffle=True, validation_split=0.1,
                            callbacks=[reduce_lr])

        score = model.evaluate(X_test, y_test, verbose=1)
        print(score)

        # 预测
        y_predict = model.predict(X_test)
        y_predict = np.rint(y_predict)
        y_test = np.rint(y_test)

        # 计算评价指标
        res = cal_metric_measure(y_test.reshape(cols_test, ), y_predict.reshape(cols_test, ))
        print('res', res)

        if res != 0:
            f1 = open(
                r'/home/ubuntu/SDP/result/CNN/KFold/3_22/3_22_calculate' + item + '_result.csv',
                'a', encoding='utf-8', newline='' "")

            csv_writer1 = csv.writer(f1)
            csv_writer1.writerow(res)
            f1.close()


if __name__ == '__main__':
    file_name_list = ["/ant-1.3","/ant-1.4","/ant-1.5","/ant-1.6","/ant-1.7","/camel-1.0","/camel-1.2","/camel-1.4","/camel-1.6", "/ivy-1.1", "/ivy-1.4", "/ivy-2.0", "/jedit-3.2", "/jedit-4.0", "/jedit-4.1", "/jedit-4.2",
                      "/jedit-4.3", "/log4j-1.0", "/log4j-1.1", "/log4j-1.2", "/lucene-2.0", "/lucene-2.2",
                      "/lucene-2.4", "/pbeans-1.0", "/pbeans-2.0", "/poi-1.5", "/poi-2.0", "/poi-2.5", "/poi-3.0",
                      "/synapse-1.0", "/synapse-1.1", "/synapse-1.2", r"/velocity-1.4", r"/velocity-1.5",
                      r"/velocity-1.6", r"/xalan-2.4", r"/xalan-2.5", r"/xalan-2.6", r"/xerces-1.2", r"/xerces-1.3",
                      r"/xerces-init"]

    for item in file_name_list:
        CNN_model(item)