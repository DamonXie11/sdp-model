import keras.regularizers
import pandas as pd
import csv as csv
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D,MaxPool1D,Embedding,LSTM,GRU
from keras.metrics import TrueNegatives,TruePositives,FalseNegatives,FalsePositives
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.regularizers import l2
from collections import Counter
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.naive_bayes import GaussianNB
import math
from xgboost import XGBClassifier

###############定义模型训练过程计算的评价值###############
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
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

# 模型学习率衰减函数
def step_decay(epoch):
    initial_lrate = 0.0015
    # drop = 0.5
    # epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) \ epochs_drop))
    return initial_lrate

# 输入两个数组，origin为实际值，predict为预测值
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


##################以下为定义的模型，模型返回值训练后的预测值#############

def xgb(X_train, y_train, X_test, y_test):
    #################xgboost###############

    xgb_model = XGBClassifier(max_depth=3,
                              learning_rate=0.1,
                              n_estimators=100,  # 使用多少个弱分类器
                              objective='binary:logistic',
                              booster='gbtree',
                              gamma=0,
                              min_child_weight=1,
                              max_delta_step=0,
                              subsample=1,
                              colsample_bytree=1,
                              reg_alpha=0,
                              reg_lambda=1,
                              seed=None  # 随机数种子
                              )

    xgb_model.fit(
        X_train,  # array, DataFrame 类型
        y_train,  # array, Series 类型
        eval_set=None,  # 用于评估的数据集，例如：[(X_train, y_train), (X_test, y_test)]
        eval_metric=None,  # 评估函数，字符串类型，例如：'mlogloss'
        early_stopping_rounds=None,
        verbose=True,  # 间隔多少次迭代输出一次信息
        xgb_model=None
    )

    preds = xgb_model.predict(X_test)

    return preds

def mlp(X_train, y_train, X_test, y_test):
    mpl_model = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    preds = mpl_model.predict(X_test)
    return preds

def DBN(X_train, y_train, X_test, y_test, input_shape):
    model = Sequential()
    # model.add(Embedding(output_dim=30, input_dim=5000, input_length=length))

    model.add(Dense(100, activation='relu', input_shape = (input_shape[0],)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), metric_F1score,
                           metric_G_measure])

    model.summary()

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

    y_predict = model.predict(X_test)
    y_predict = np.rint(y_predict)

    return y_predict

def Li_CNN(X_train, y_train, X_test, y_test, input_shape):

    num_epochs = 50
    batch_size = 128


    model = Sequential()
    model.add(Embedding(output_dim=30, input_dim=X_train.shape[0], input_length=X_train.shape[1]))
    model.add(Conv1D(10, kernel_size=5,activation='relu', input_shape=input_shape))
    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), metric_F1score,
                           metric_G_measure])

    model.summary()

    reduce_lr = LearningRateScheduler(step_decay)

    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test, y_test))

    # 预测
    y_predict = model.predict(X_test)
    y_test = np.rint(y_predict)

    return y_predict

def CNN_GRU(X_train, y_train, X_test, y_test, input_shape):
    num_epochs = 50
    batch_size = 128

    model = Sequential()
    model.add(Embedding(output_dim=30, input_dim=X_train.shape[0], input_length=X_train.shape[1],
                        embeddings_regularizer=keras.regularizers.l2(0.1)))
    model.add(Conv1D(10, kernel_size=5, activation='relu', kernel_initializer="he_normal"))
    model.add(MaxPool1D(pool_size=2))

    model.add(GRU(10, dropout=0.3, activation='relu', return_sequences=True))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', kernel_initializer="he_normal"))
    model.add(Dropout(0.5))

    # 编译模型
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), metric_F1score,
                           metric_G_measure])

    model.summary()

    reduce_lr = LearningRateScheduler(step_decay)

    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_split=0.1,
                        callbacks=[reduce_lr])

    score = model.evaluate(X_test, y_test, verbose=1)
    print(score)

    # 预测
    y_predict = model.predict(X_test)
    y_predict = np.rint(y_predict)

    return y_predict

###################以下为不同验证方法###########

# 跨版本预测
def CrossVersion(train, test, name):
    # In[2]:

    data_train = pd.read_csv(
        r'D:\Software_fault_prediction-main\Deep_learning_regression\datasets\mapped data' + train + '.csv')

    data_test = pd.read_csv(
        r'D:\Software_fault_prediction-main\Deep_learning_regression\datasets\mapped data' + test + '.csv')

    # 将有bug的改为一致的
    for i in range(len(data_train['bugs'])):
        if data_train['bugs'][i] > 1:
            data_train['bugs'][i] = 1

    data_train = data_train.drop(['filename'], axis=1)


    X_train = data_train.drop('bugs', axis=1)
    y_train = pd.DataFrame(data_train['bugs'], columns=['bugs'])

    # 将有bug的改为一致的
    for i in range(len(data_test['bugs'])):
        if data_test['bugs'][i] > 1:
            data_test['bugs'][i] = 1

    data_test = data_test.drop(['filename'], axis=1)

    X_test = data_test.drop('bugs', axis=1)
    y_test = pd.DataFrame(data_test['bugs'], columns=['bugs'])

    # In[4]:

    # 仅在训练集中使用随机过采样
    ros = RandomOverSampler(random_state=None)
    X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)

    # In[5]:

    length = X_train.shape[1]
    cols_test = X_test.shape[0]
    input_shape = (X_train.shape[1], 1)

    # In[6]:

    # 归一化
    scaler = MinMaxScaler()
    MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train = MinMaxScaler().fit_transform(X_train_overSampling)
    X_test = MinMaxScaler().fit_transform(X_test)

    y_train = y_train_overSampling.values.reshape(y_train_overSampling.shape[0], 1)
    X_train_re = X_train.reshape(X_train.shape[0], length, 1)
    X_test_re = X_test.reshape(X_test.shape[0], length, 1)
    y_train = MinMaxScaler().fit_transform(y_train)


    # In[7]:

    # In[28]:

    for i in range(1):

        ##############在此处修改模型#####################
        y_predict = mlp(X_train, y_train, X_test, y_test)

        # 计算评价指标
        res = cal_metric_measure(y_test.values.reshape(cols_test, ), y_predict.reshape(cols_test, ))
        print('res', res)

        if res != 0:
            # 计算手工计算结果计算储存路径
            f1 = open(
                r'D:\Software_fault_prediction-main\result\MLP\MLP_CrossVersion\\' + name + '_result.csv', 'a', encoding='utf-8', newline='' "")

            csv_writer1 = csv.writer(f1)

            csv_writer1.writerow(res)
            f1.close()

# 10Fold版本内预测
def KFold(item):
    data_all = pd.read_csv(
        r'D:\Software_fault_prediction-main\Deep_learning_regression\datasets\mapped data' + item + '.csv')

    # 将有bug的改为一致的
    for i in range(len(data_all['bugs'])):
        if data_all['bugs'][i] > 1:
            data_all['bugs'][i] = 1

    data = data_all.drop(['filename'], axis=1)

    X_data = data.drop('bugs', axis=1)
    y_data = pd.DataFrame(data['bugs'], columns=['bugs'])

    # KFold

    all_F_measure = []
    num_epochs = 50

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in skf.split(X_data.values, y_data.values):
        # print('train_index', train_index)
        # print('X_train', X_train.shape)
        X_train, X_test = X_data.values[train_index], X_data.values[test_index]
        y_train, y_test = y_data.values[train_index], y_data.values[test_index]

        # 仅在训练集中使用随机过采样
        ros = RandomOverSampler(random_state=0)
        X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)
        # 先将X_train和y_train重新拼接在一起
        # data_train_overSampling = pd.concat([X_train_overSampling.,y_train_overSampling],axis=1)
        # data_train_overSampling

        length = X_train.shape[1]
        cols_test = X_test.shape[0]
        input_shape = (X_train.shape[1], 1)

        # 归一化
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train_overSampling)
        X_test = MinMaxScaler().fit_transform(X_test)

        y_train = y_train_overSampling.reshape(y_train_overSampling.shape[0], 1)
        X_train_re = X_train.reshape(X_train.shape[0], length, 1)
        X_test_re = X_test.reshape(X_test.shape[0], length, 1)
        y_train = MinMaxScaler().fit_transform(y_train)

        ##############在此处修改模型##################
        y_predict = DBN(X_train, y_train, X_test, y_test,input_shape)

        # 计算评价指标
        res = cal_metric_measure(y_test.reshape(cols_test, ), y_predict.reshape(cols_test, ))
        print('res', res)

        if res != 0:
            # 计算手工计算结果计算储存路径
            f1 = open(
                r'D:\Software_fault_prediction-main\result\DBN\DBN_Kfold\\' + item + '_result.csv', 'a',
                encoding='utf-8', newline='' "")

            csv_writer1 = csv.writer(f1)

            # csv_writer.writerow(["Y","Y_predict"])

            csv_writer1.writerow(res)
            f1.close()

# 手工特征30次重复实验预测
def holdout30(item):
    data_all = pd.read_csv(
        'D:\Software_fault_prediction-main\Deep_learning_regression\datasets\labeled data' + item + '.csv')

    # 将有bug的改为一致的
    for i in range(len(data_all['bug'])):
        if data_all['bug'][i] > 1:
            data_all['bug'][i] = 1

    data = data_all.drop(['version'], axis=1)
    data = data.drop(['name1'], axis=1)
    data = data.drop(['name'], axis=1)

    X_data = data.drop('bug', axis=1)
    y_data = pd.DataFrame(data['bug'], columns=['bug'])

    # KFold

    all_F_measure = []
    num_epochs = 50

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for index in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size=0.2, random_state=None, stratify=y_data.values)

        # 仅在训练集中使用随机过采样
        ros = RandomOverSampler(random_state=0)
        X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)
        # 先将X_train和y_train重新拼接在一起
        # data_train_overSampling = pd.concat([X_train_overSampling.,y_train_overSampling],axis=1)
        # data_train_overSampling

        length = X_train.shape[1]
        cols_test = X_test.shape[0]
        input_shape = (X_train.shape[1], 1)

        # 归一化
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train_overSampling)
        X_test = MinMaxScaler().fit_transform(X_test)

        y_train = y_train_overSampling.reshape(y_train_overSampling.shape[0], 1)
        X_train_re = X_train.reshape(X_train.shape[0], length, 1)
        X_test_re = X_test.reshape(X_test.shape[0], length, 1)
        y_train = MinMaxScaler().fit_transform(y_train)

        ###############在此处修改模型###############
        y_predict = DBN(X_train, y_train, X_test, y_test,input_shape)

        # 计算评价指标
        res = cal_metric_measure(y_test.reshape(cols_test, ), y_predict.reshape(cols_test, ))

        if res != 0:
            # 计算手工计算结果计算储存路径
            f1 = open(
                r'D:\Software_fault_prediction-main\result\DBN\DBN_30holdout\\' + item + '_result.csv', 'a', encoding='utf-8', newline='' "")

            csv_writer1 = csv.writer(f1)

            csv_writer1.writerow(res)
            f1.close()

# 手工特征根据特征相关性筛选后的30次重复实验预测
def holdout30_selected(item):
    data_all = pd.read_csv(
        'D:\Software_fault_prediction-main\Deep_learning_regression\datasets\labeled data' + item + '.csv')

    # 将有bug的改为一致的
    for i in range(len(data_all['bug'])):
        if data_all['bug'][i] > 1:
            data_all['bug'][i] = 1

    data = data_all.drop(['version'], axis=1)
    data = data.drop(['name1'], axis=1)
    data = data.drop(['name'], axis=1)



    X_data = data.drop('bug', axis=1)
    y_data = pd.DataFrame(data['bug'], columns=['bug'])

    correlation_matrix = X_data.corr()
    print(correlation_matrix)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

    correlation_matrix_ut = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape)).astype(np.bool_))
    correlation_matrix_melted = correlation_matrix_ut.stack().reset_index()
    correlation_matrix_melted.columns = ['word1', 'word2', 'correlation']
    res = correlation_matrix_melted[(correlation_matrix_melted['word1'] != \
                                     correlation_matrix_melted['word2']) & (
                                            correlation_matrix_melted['correlation'] > .85)]

    dropped_name = res.drop(['correlation'], axis=1)

    dropped_name = dropped_name.values.flatten()
    dropped_name = list(set(dropped_name))

    X_data.drop(dropped_name, axis=1)

    # KFold

    all_F_measure = []
    num_epochs = 50

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for index in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size=0.2, random_state=None, stratify=y_data.values)

        # 仅在训练集中使用随机过采样
        ros = RandomOverSampler(random_state=0)
        X_train_overSampling, y_train_overSampling = ros.fit_resample(X_train, y_train)
        # 先将X_train和y_train重新拼接在一起
        # data_train_overSampling = pd.concat([X_train_overSampling.,y_train_overSampling],axis=1)
        # data_train_overSampling

        length = X_train.shape[1]
        cols_test = X_test.shape[0]
        input_shape = (X_train.shape[1], 1)

        # 归一化
        scaler = MinMaxScaler()
        MinMaxScaler(copy=True, feature_range=(0, 1))
        X_train = MinMaxScaler().fit_transform(X_train_overSampling)
        X_test = MinMaxScaler().fit_transform(X_test)

        y_train = y_train_overSampling.reshape(y_train_overSampling.shape[0], 1)
        X_train_re = X_train.reshape(X_train.shape[0], length, 1)
        X_test_re = X_test.reshape(X_test.shape[0], length, 1)
        y_train = MinMaxScaler().fit_transform(y_train)

        #############在此处修改模型############
        y_predict = mlp(X_train, y_train, X_test, y_test)

        # 计算评价指标
        res = cal_metric_measure(y_test.reshape(cols_test, ), y_predict.reshape(cols_test, ))

        if res != 0:
            # 计算手工计算结果计算储存路径
            f1 = open(
                r'D:\Software_fault_prediction-main\result\MLP\MLP_30holdout_selected' + item + '_result.csv', 'a', encoding='utf-8', newline='' "")

            csv_writer1 = csv.writer(f1)

            csv_writer1.writerow(res)
            f1.close()


if __name__ == '__main__':

    #########cross version使用的数据##########
    # file_list = [[r'\camel-1.4', r'\camel-1.6', 'camel'],[r'\jedit-4.0', r'\jedit-4.1', 'jedit'],[r'\lucene-2.0', r'\lucene-2.2', 'lucene'], [r'\xalan-2.5', r'\xalan-2.6', 'xalan'], [r'\xerces-1.2', r'\xerces-1.3', 'xerces'], [r'\synapse-1.1', r'\synapse-1.2', 'synapse'], [r'\poi-2.5', r'\poi-3.0', 'poi']]

    ##############KFold使用的数据###############
    file_list = ["/ant-1.3","/ant-1.4","/ant-1.5","/ant-1.6","/ant-1.7","/camel-1.0", "/camel-1.2", "/camel-1.4", "/camel-1.6", "/ivy-1.1", "/ivy-1.4", "/ivy-2.0",
                      "/jedit-3.2", "/jedit-4.0", "/jedit-4.1", "/jedit-4.2",
                      "/jedit-4.3", "/log4j-1.0", "/log4j-1.1", "/log4j-1.2", "/lucene-2.0", "/lucene-2.2",
                      "/lucene-2.4", "/pbeans-1.0", "/pbeans-2.0", "/poi-1.5", "/poi-2.0", "/poi-2.5", "/poi-3.0",
                      "/synapse-1.0", "/synapse-1.1", "/synapse-1.2", r"/velocity-1.4", r"/velocity-1.5",
                      r"/velocity-1.6", r"/xalan-2.4", r"/xalan-2.5", r"/xalan-2.6", r"/xerces-1.2", r"/xerces-1.3",
                      r"/xerces-init"]

    # file_list = [[r'\xalan-2.5', r'\xalan-2.6', 'xalan'], [r'\xerces-1.2', r'\xerces-1.3', 'xerces'], [r'\synapse-1.1', r'\synapse-1.2', 'synapse'], [r'\poi-2.5', r'\poi-3.0', 'poi']]

    # file_list = [[r'\ant-1.4', r'\ant-1.5', 'ant']]

    ##############holdout30、holdout30_selected手工特征使用的数据####################
    # file_list = [r"\camel", r"\jedit", r"\lucene", r"\poi", r"\synapse", r"\xalan", r"\xerces"]

    for item in file_list:
        ##########修改预测方法############

        # CrossVersion(item[0], item[1], item[2])
        KFold(item)
        # holdout30(item)
        # holdout30_selected(item)