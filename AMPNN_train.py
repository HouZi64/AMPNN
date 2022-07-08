# -*- coding: utf-8 -*-
# @Time : 2021/4/16 17:02
# @Author : Mingzheng
import sys
import os

import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


import random
from datetime import datetime
from tools import tool

from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, SpatialDropout2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from models.network.AMPNN import ampnn

from keras import Input,Model
# Define general variables
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set your path to the dataset

us8k_path = os.path.abspath(r'Rainfall_Audio_XZ(RA_XZ)_dataset')
audio_path = os.path.join(us8k_path, 'data_nobackground_split_balance')
metadata_path = os.path.join(us8k_path, 'metadata_nobackground_split_balance\metadata.csv')
models_path = os.path.abspath('./models')
data_path = os.path.abspath('./data')

# Ensure "channel last" data format on Keras
keras_backend.set_image_data_format('channels_last')

# Define a labels array for future use
labels = [
        'small', 'middle', 'heavy', 'violent', 'no_rain'
    ]

import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
keras_backend.set_session(tf.Session(config=config))

# Pre-processed MFCC coefficients
X_mfcc = np.load(r"data\rain\feature/X-nobackground-balance-mfcc.npy",allow_pickle=True)
y_mfcc = np.load(r"data\rain\feature/y-nobackground-balance-mfcc.npy",allow_pickle=True)
X_mel = np.load(r"data\rain\feature/X-nobackground-balance-mel.npy",allow_pickle=True)
y_mel = np.load(r"data\rain\feature/y-nobackground-balance-mel.npy",allow_pickle=True)


# Metadata
metadata = pd.read_csv(metadata_path)

indexes = []
total = len(metadata)
indexes = list(range(0, total))

# Randomize indexes
random.shuffle(indexes)

# Divide the indexes into Train and Test
# test_split_pct = 3000
# split_offset = math.floor(test_split_pct * total / 100)
split_offset = 1400
# Split the metadata
test_split_idx = indexes[0:split_offset]
train_split_idx = indexes[split_offset:total]

# How data should be structured
num_rows = 40
num_columns = 173
num_channels = 1
# Split the features with the same indexes
X_mfcc_test = np.take(X_mfcc, test_split_idx, axis=0)
y_mfcc_test = np.take(y_mfcc, test_split_idx, axis=0)
X_mfcc_train = np.take(X_mfcc, train_split_idx, axis=0)
y_mfcc_train = np.take(y_mfcc, train_split_idx, axis=0)
X_mel_test = np.take(X_mel, test_split_idx, axis=0)
y_mel_test = np.take(y_mel, test_split_idx, axis=0)
X_mel_train = np.take(X_mel, train_split_idx, axis=0)
y_mel_train = np.take(y_mel, train_split_idx, axis=0)

X_mfcc_test = X_mfcc_test.reshape(X_mfcc_test.shape[0],num_rows, num_columns, num_channels)
X_mfcc_train = X_mfcc_train.reshape(X_mfcc_train.shape[0],num_rows, num_columns, num_channels)


# Also split metadata
test_meta = metadata.iloc[test_split_idx]
train_meta = metadata.iloc[train_split_idx]

# Print status
print("Test split: {} \t\t Train split: {}".format(len(test_meta), len(train_meta)))
print("X test shape: {} \t X train shape: {}".format(X_mfcc_test.shape, X_mfcc_train.shape))
print("y test shape: {} \t\t y train shape: {}".format(y_mfcc_test.shape, y_mfcc_train.shape))
print("X test shape: {} \t X train shape: {}".format(X_mel_test.shape, X_mel_train.shape))
print("y test shape: {} \t\t y train shape: {}".format(y_mel_test.shape, y_mel_train.shape))
le = LabelEncoder()
y_mfcc_test_encoded = to_categorical(le.fit_transform(y_mfcc_test))
y_mfcc_train_encoded = to_categorical(le.fit_transform(y_mfcc_train))
y_raw_test_encoded = to_categorical(le.fit_transform(y_mel_test))
y_raw_train_encoded = to_categorical(le.fit_transform(y_mel_train))

y_mfcc_train_encoded_lda = np.where(y_mfcc_train_encoded==1)[-1]

# Total number of labels to predict (equal to the network output nodes)
num_labels = y_raw_train_encoded.shape[1]

# Regularization rates
spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005
from keras.callbacks import LearningRateScheduler

# learning rate decay
def scheduler(epoch):
    if epoch% 50 == 0 and epoch != 0:
        lr= K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr* 0.1)
        print("lr changed to {}".format(lr* 0.1))
    return K.get_value(model.optimizer.lr)

learning_rate_callback = LearningRateScheduler(scheduler)
# larning rate monitoring
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
model = model_mz(5).create_model()
adam = Adam(lr=1e-3, beta_1=0.99, beta_2=0.999,decay=1e-3)
lr_metric = get_lr_metric(adam)

model.compile(optimizer=adam,
              loss = 'categorical_crossentropy',
              metrics=['accuracy',lr_metric])
model.summary()
num_epochs = 100
num_batch_size = 128
# model_file = 'test.hdf5'
model_file = 'test3.ckpt'
model_path = os.path.join(models_path, model_file)

# Save checkpoints
checkpointer = ModelCheckpoint(filepath=model_path,
                               verbose=1,
                               save_weights_only=True,
                               save_best_only=True)
start = datetime.now()
print('------------------------------begin training------------------------------')
history = model.fit([X_mfcc_train,X_mel_train],
                    y_raw_train_encoded,
                    batch_size=num_batch_size,
                    epochs=num_epochs,
                    validation_split=1/12.,
                    callbacks=[checkpointer,learning_rate_callback],
                    verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)
# Load best saved model
model.load_weights(model_path)

tool.model_evaluation_report(model,[X_mfcc_train,X_mel_train], y_raw_train_encoded,
                                [X_mfcc_test,X_mel_test], y_raw_test_encoded)
tool.plot_train_history(history, x_ticks_vertical=True)

# Predict probabilities for test set
y_probs = model.predict( [X_mfcc_test,X_mel_test], verbose=0)

# Get predicted labels
# y_probs = y_probs['outputs']
yhat_probs = np.argmax(y_probs, axis=1)
y_trues = np.argmax(y_raw_test_encoded, axis=1)

# Add "pred" column
test_meta['pred'] = yhat_probs

# Sets decimal precision (for printing output only)
np.set_printoptions(precision=2)

# Compute confusion matrix data
cm = confusion_matrix(y_trues, yhat_probs)

tool.plot_confusion_matrix(cm,
                          labels,
                          normalized=False,
                          title="Model Performance",
                          cmap=plt.cm.Blues,
                          size=(12,12))

# Find per-class accuracy from the confusion matrix data
accuracies = tool.acc_per_class(cm)

pd.DataFrame({
    'CLASS': labels,
    'ACCURACY': accuracies
}).sort_values(by="ACCURACY", ascending=False)

# Build classification report
re = classification_report(y_trues, yhat_probs, labels=[0,1,2,3,4], target_names=labels)


print(re)
