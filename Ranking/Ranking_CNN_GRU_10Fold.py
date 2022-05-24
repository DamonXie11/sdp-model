import keras.regularizers
import pandas as pd
import csv as csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LassoCV
import numpy as np
import time
import math
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, Conv1D, Flatten, MaxPool2D, MaxPool1D, Embedding, GRU, LSTM, SimpleRNN
from keras.metrics import TrueNegatives, TruePositives, FalseNegatives, FalsePositives
import keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.regularizers import l2
from collections import Counter
from keras.callbacks import LearningRateScheduler
from sklearn.naive_bayes import GaussianNB
import math


def lasso_model():
    lasso = LassoCV()
    return lasso


def processing(item):
    data_all = pd.read_csv(
        r'/home/ubuntu/SDP/datasets/mapped data' + item + '.csv')

    data = data_all.drop(['filename'], axis=1)

    X_data = data.drop('bugs', axis=1)
    y_data = pd.DataFrame(data['bugs'], columns=['bugs'])

    kf = KFold(n_splits=10, shuffle=True, random_state = 0)

    flag = 0
    for train_index, test_index in kf.split(X_data):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        # 仅在训练集中使用随机过采样
        ros = RandomOverSampler(random_state=0)
        X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)

        # 归一化
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)

        model = lasso_model()

        model.fit(X_train, y_train.values.flatten())

        res = model.predict(X_test)

        predictions_y_round = np.rint(res)

        f = open(r'/home/ubuntu/SDP/result/Ranking/lasso' + item + '.csv', 'a+', encoding='utf-8', newline='')

        f.writelines('file_index,original_value,predict_value\n')

        for item_origin, item_predict in zip(y_test.values.flatten(), predictions_y_round):
            print(item_origin)
            print(item_predict)
            f.writelines(str(flag) + ',' + str(item_origin) + ',' + str(item_predict) + '\n')
            flag += 1

        f.close()


def CNN_GRU_model(embedding_input_dim, length, input_shape):
    model = Sequential()
    model.add(Embedding(output_dim=30, input_dim=embedding_input_dim, input_length=length,
                        embeddings_regularizer=keras.regularizers.l2(0.1)))

    model.add(Conv1D(10, kernel_size=5, activation='relu', kernel_initializer="he_normal",
                     input_shape=input_shape))
    model.add(MaxPool1D(pool_size=2))

    model.add(GRU(32, dropout=0.5, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.3))

    # model.add(Dense(80, activation='relu', kernel_initializer="he_normal"))

    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    model.summary()

    return model


def processing_NN(item):
    data_all = pd.read_csv(
        r'/home/ubuntu/SDP/datasets/mapped data' + item + '.csv')

    data = data_all.drop(['filename'], axis=1)

    X_data = data.drop('bugs', axis=1)
    y_data = pd.DataFrame(data['bugs'], columns=['bugs'])

    kf = KFold(n_splits=10, shuffle=True)

    flag = 0
    for train_index, test_index in kf.split(X_data):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        # 归一化
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)

        length = X_train.shape[1]
        cols_test = X_test.shape[0]
        input_shape = (X_train.shape[1], 1)

        model = CNN_GRU_model(X_train.shape[0], length, input_shape)

        # 设置早停
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        history = model.fit(X_train, y_train.values.flatten(), epochs=50, batch_size=32, shuffle=True,
                            validation_split=0.2)

        res = model.predict(X_test)

        predictions_y_round = np.rint(res)

        f = open(r'/home/ubuntu/SDP/result/Ranking/CNN_GRU_5_17_2' + item + '.csv', 'a+', encoding='utf-8', newline='')

        if flag == 0:
            f.writelines('file_index,original_value,predict_value\n')

        for item_origin, item_predict in zip(y_test.values.flatten(), predictions_y_round.flatten()):
            print(item_origin)
            print(item_predict)
            f.writelines(str(flag) + ',' + str(item_origin) + ',' + str(item_predict) + '\n')
            flag += 1

        f.close()


if __name__ == '__main__':
    file_name_list = ["/ant-1.3","/ant-1.4","/ant-1.5","/ant-1.6","/ant-1.7","/camel-1.0","/camel-1.2","/camel-1.4","/camel-1.6", "/ivy-1.1", "/ivy-1.4", "/ivy-2.0", "/jedit-3.2", "/jedit-4.0", "/jedit-4.1", "/jedit-4.2",
                      "/jedit-4.3", "/log4j-1.0", "/log4j-1.1", "/log4j-1.2", "/lucene-2.0", "/lucene-2.2",
                      "/lucene-2.4", "/pbeans-1.0", "/pbeans-2.0", "/poi-1.5", "/poi-2.0", "/poi-2.5", "/poi-3.0",
                      "/synapse-1.0", "/synapse-1.1", "/synapse-1.2", r"/velocity-1.4", r"/velocity-1.5",
                      r"/velocity-1.6", r"/xalan-2.4", r"/xalan-2.5", r"/xalan-2.6", r"/xerces-1.2", r"/xerces-1.3",
                      r"/xerces-init"]

    # file_name_list = ["/poi-1.5", "/poi-2.0", "/poi-2.5", "/poi-3.0",
    #                   "/synapse-1.0", "/synapse-1.1", "/synapse-1.2", r"/velocity-1.4", r"/velocity-1.5",
    #                   r"/velocity-1.6", r"/xalan-2.4", r"/xalan-2.5", r"/xalan-2.6", r"/xerces-1.2", r"/xerces-1.3",
    #                   r"/xerces-init"]

    # file_name_list = [r"\ant-1.3"]
    f1_list = []

    # file_res = open(
    #     r'/home/ubuntu/SDP/result/NN/2_16/NN_average_2_16.csv',
    #     'a+')

    for item in file_name_list:
        res = processing_NN(item)
