import numpy as np
import os

# read data
print("Reading data...")
ucl_har_path = 'UCI HAR Dataset'


training_records = 7352
test_records = 2947
def read_ucl_har(path):
    # training set
    workspace = os.getcwd()
    os.chdir(path+'/train/Inertial Signals')
    raw_x_files = os.listdir()
    tmp=[]
    for file in raw_x_files:
        with open(file) as f:
            data=f.read()
        data=data.strip().split()
        tmp.append(np.array(data,dtype=float).reshape(training_records,-1,1))
    x_train = tmp[0]
    for i,_ in enumerate(tmp):
        if i == 0:
            continue
        x_train = np.concatenate((x_train,tmp[i]),axis=2)
    os.chdir(workspace)

    with open(path+'/train/y_train.txt') as f:
        data = f.read()
    data=data.strip().split()
    y_train = np.array(data,dtype=int).reshape(-1,1)

    with open(path+'/train/subject_train.txt') as f:
        data = f.read()
    data=data.strip().split()
    s_train = np.array(data,dtype=int)

    # test set
    os.chdir(path+'/test/Inertial Signals')
    raw_x_files = os.listdir()
    tmp=[]
    for file in raw_x_files:
        with open(file) as f:
            data=f.read()
        data=data.strip().split()
        tmp.append(np.array(data,dtype=float).reshape(test_records,-1,1))
    x_test = tmp[0]
    for i,_ in enumerate(tmp):
        if i == 0:
            continue
        x_test = np.concatenate((x_test,tmp[i]),axis=2)
    os.chdir(workspace)

    with open(path+'/test/y_test.txt') as f:
        data = f.read()
    data=data.strip().split()
    y_test = np.array(data,dtype=int).reshape(-1,1)

    with open(path+'/test/subject_test.txt') as f:
        data = f.read()
    data=data.strip().split()
    s_test = np.array(data,dtype=int)

    with open(path+'/features.txt') as f:
        data = f.read()
    data=data.strip().split()
    feature_names = np.array(data[1::2])

    return x_train, y_train, s_train, x_test, y_test, s_test, feature_names

raw_x_train, raw_y_train, s_train, raw_x_test, raw_y_test, s_test, feature_names = read_ucl_har(ucl_har_path)

# preprocess
from keras.utils import to_categorical
print('Preprocessing...')
# 1. LINEAR INTERPOLATION
print('>> 1. Linear interpolation')

# 2. SCALING AND NORMALIZATION
print('>> 2. scaling and normalization')

from sklearn.preprocessing import MinMaxScaler
tmp_x_train = MinMaxScaler().fit(raw_x_train.reshape(-1,9)).transform(raw_x_train.reshape(-1,9)).reshape(training_records,-1,9)
tmp_x_test = MinMaxScaler().fit(raw_x_test.reshape(-1,9)).transform(raw_x_test.reshape(-1,9)).reshape(test_records,-1,9)
tmp_y_train = raw_y_train
tmp_y_test = raw_y_test

# 3. SEGMENTATION
# from scipy.stats import mode
print('>> 3. segmentation')
# window_size = 128
# overlap_rate = 0.5
# print('window_size={0} overlap_rate={1}'.format(window_size,overlap_rate))

# def aggreation(dataset, window_size, overlap_rate):
#     split_num = np.arange(0,len(dataset)-window_size,window_size*overlap_rate,dtype=int)
#     if len(dataset)-window_size not in split_num:
#         split_num = np.append(split_num,len(dataset)-window_size)
#     l = split_num[0]
#     ret = np.array([dataset[l:l+window_size]])
#     for i in split_num[split_num<=len(dataset)-window_size]:
#         if i == 0:
#             continue
#         ret = np.concatenate((ret,np.array([dataset[i:i+window_size]])),axis=0)
#     return ret

# def y_to_categorical(dataset):
#     l1= dataset.shape[0]
#     tmp = dataset.reshape(l1,-1)
#     ret=[]
#     for v in tmp:
#         ret.append(mode(v,keepdims=True).mode)
#     ret = to_categorical(np.array(ret,dtype=int)-1)
#     return ret

# training set
# print('size of training set:{0}'.format(len(tmp_x_train)),tmp_x_train.shape)
# x_train = aggreation(tmp_x_train,window_size,overlap_rate)
# y_train = aggreation(tmp_y_train,window_size,overlap_rate)
# y_train = y_to_categorical(y_train)

x_train = tmp_x_train
y_train = to_categorical((tmp_y_train-1).reshape(-1))
print('x_train:',x_train.shape)
print('y_train:',y_train.shape)

# test set
# print('size of test set:{0}'.format(len(tmp_x_test)),tmp_x_test.shape)
# x_test = aggreation(tmp_x_test,window_size,overlap_rate)
# y_test = aggreation(tmp_y_test,window_size,overlap_rate)
# y_test = y_to_categorical(y_test)

x_test = tmp_x_test
y_test = to_categorical((tmp_y_test-1).reshape(-1))
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)

# # LSTM-CNN
from keras.models import Sequential
from keras.layers import LSTM,Conv1D,MaxPooling1D,GlobalAveragePooling1D,BatchNormalization,Dense,Reshape
from keras.optimizers import Adam

input_shape=x_train.shape[1:]
print('input shape of LSTM-CNN is',input_shape)

model = Sequential()

model.add(LSTM(32,activation='relu',input_shape=input_shape,return_sequences=True,use_bias=False))
model.add(LSTM(32,activation='relu',use_bias=True,return_sequences=True))
model.add(Reshape((1,x_train.shape[1],-1)))

model.add(Conv1D(64,kernel_size=5,strides=2,activation='relu'))
model.add(Reshape((-1,64)))
model.add(MaxPooling1D(pool_size=2,strides=2))
model.add(Conv1D(128,kernel_size=3,strides=1,activation='relu'))

model.add(GlobalAveragePooling1D())
model.add(BatchNormalization())

model.add(Dense(6,activation='softmax'))

model.summary()

# hyper-parameters
batch_size=192
epoch=200
learning_rate=0.001
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=learning_rate),metrics=['accuracy','F1Score'])
history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch)

# evaluation and visualization
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

y_pred_train = np.argmax(model.predict(x_train),axis=1)
print(classification_report(np.argmax(y_train,axis=1), y_pred_train))

y_pred_test = np.argmax(model.predict(x_test),axis=1)
print(classification_report(np.argmax(y_test,axis=1), y_pred_test))

score = model.evaluate(x_test,y_test)

from pycm import ConfusionMatrix
print(ConfusionMatrix(actual_vector=np.argmax(y_train,axis=1), predict_vector=y_pred_train))
print(ConfusionMatrix(actual_vector=np.argmax(y_test,axis=1), predict_vector=y_pred_test))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], '', label='accuracy') 
plt.plot(history.history['loss'], '--', label='loss') 
plt.title('Training Accuracy and Loss') 
plt.ylabel('Accuracy and Loss') 
plt.xlabel('Training Epoch') 
plt.ylim(0) 
plt.legend() 
plt.show()