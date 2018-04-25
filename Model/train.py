"""
p, r, f1: 0.184, 0.186,  0.185;  wt sensitive: 1:20, patience = 3
p, r, f1 : 0.225 0.206 0.215; wt sensitive: 1:10, patience = 5
p, r, f1: 0.294 0.186 0.228;  wt sensitive: 1:20, patience = 5
0.365 0.141 0.204;  wt sensitive: 1:20, patience = 5, dropout = 0.3
"""


import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Concatenate, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM
from keras import backend as K

from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, confusion_matrix)
import sklearn.metrics as sklm

import pickle
import os
import matplotlib.pyplot as plt

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 106
batch_size = 32

filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 30

Q_LEN = 29
ANS_LEN = 286 


print("Loading the Embedding matrix:")
embedding_matrix = pickle.load(open('../Data/data_abhishek/embedding_glove', "rb"))
num_words = embedding_matrix.shape[0] -1

print("Loading the data")

ques_train = np.load('../Data2/x_train_q')
ques_test = np.load('../Data2/x_test_q')

answer_train = np.load('../Data2/x_train_a') 
answer_test = np.load('../Data2/x_test_a')

labels_test = to_categorical(pickle.load(open('../Data2/label_test.p', 'rb')))
labels_train = to_categorical(pickle.load(open('../Data2/label_train.p', 'rb')))

print("Printing the dimensions:")
print(ques_train.shape , ques_test.shape, answer_train.shape, answer_test.shape,  labels_train.shape, labels_test.shape)
print("********")

# Designing the model

embedding_layer_ques = Embedding(num_words+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=Q_LEN, trainable=False, name = "embedding_layer_ques")
embedding_layer_ans = Embedding(num_words+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=ANS_LEN, trainable=False, name = "embedding_layer_ans")

ques_branch = Sequential()
ques_branch.add(embedding_layer_ques)
ques_branch.add(Dropout(0.3))
ques_branch.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
ques_branch.add(GlobalMaxPooling1D())


answer_branch  = Sequential()
answer_branch.add(embedding_layer_ans)
answer_branch.add(Dropout(0.3))
answer_branch.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
answer_branch.add(GlobalMaxPooling1D())


model = Sequential()
model.add(Merge([ques_branch, answer_branch], mode = 'concat'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
filepath = './saved_models/selftrain_hybrid_latest_ckpt.hdf5'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
class_weight = {0 : 1., 1: 20.}

history = model.fit({'embedding_layer_ques_input': ques_train, 'embedding_layer_ans_input':answer_train}, labels_train, 
          batch_size=batch_size, 
          epochs=epochs,
          validation_data=({'embedding_layer_ques_input': ques_test, 'embedding_layer_ans_input':answer_test}, labels_test), class_weight = class_weight, callbacks=[earlyStopping], verbose =1, shuffle = True)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = model.predict_classes({'embedding_layer_ques_input': ques_test, 'embedding_layer_ans_input':answer_test})
y_prob = np.asarray(model.predict({'embedding_layer_ques_input': ques_test, 'embedding_layer_ans_input':answer_test}))[:,1]
y_test_prime = np.argmax(labels_test,1)
accuracy = accuracy_score(y_test_prime, y_pred)
recall = recall_score(y_test_prime, y_pred)
precision = precision_score(y_test_prime, y_pred)
f1 = f1_score(y_test_prime, y_pred)
print("Final metrics: ")
print(recall, precision, f1)



