import sys
import os
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
from datetime import datetime

from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, SpatialDropout2D, Activation, Conv2D, MaxPooling2D,\
    BatchNormalization, GlobalAveragePooling2D, LeakyReLU,Flatten,LSTM,Reshape,Dropout,MaxPooling1D,\
    Concatenate,Permute,merge,Add,GlobalMaxPooling2D,multiply,GlobalAveragePooling1D,GlobalMaxPooling1D

from keras.layers import Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras import Model
from keras.regularizers import  l2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras import Input
keras_backend.set_image_data_format('channels_last')

class model_mz():
    def __init__(self,output_dim):
        self.l2_rate = 0.5
        self.spatial_dropout_rate_1 = 0.5
        self.numlabels = 10
        self.output_dim = output_dim
        # cnn
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate), activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate) )
        self.conv3 = Conv2D(filters=256, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate) )
        self.conv4 = Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate) )
        self.conv5 = Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate))
        self.leakyRelu = LeakyReLU(alpha=0.1)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.bn5 = BatchNormalization()


        self.sd = SpatialDropout2D(self.spatial_dropout_rate_1)

        self.maxpool = MaxPooling2D(pool_size=(3, 3),padding='same')
        self.gap = GlobalAveragePooling2D()
        self.dense4 = Dense(64, activation='relu')
        self.conv2DTranspose1 = Conv2DTranspose(filters=128,kernel_size=(3,3),kernel_regularizer=l2(self.l2_rate),activation='relu')
        self.conv2DTranspose2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate),
                                                activation='relu')
        self.conv2DTranspose3 = Conv2DTranspose(filters=40, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate),
                                                activation='relu')
        # self.dense = Dense(self.numlabels,activation='softmax')
        # lstm
        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate), activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(3, 3))
        self.conv7 = Conv2D(filters=256, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate), activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(3, 3))
        self.reshape = Reshape((-1, 88200))
        self.lstm1 = LSTM(64, return_sequences=True)
        self.dropout1 = Dropout(0.5)
        self.lstm2 = LSTM(128, return_sequences=True)
        self.dropout2 = Dropout(0.5)
        self.lstm3 = LSTM(64)
        self.dropout3 = Dropout(0.5)

        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dropout4 = Dropout(0.5)
        self.dense2 = Dense(128, activation='relu')
        self.dropout5 = Dropout(0.5)
        self.dense3 = Dense(64, activation='sigmoid')
        self.dropout6 = Dropout(0.5)
        self.bn6 = BatchNormalization()
        self.bn7 = BatchNormalization()
        self.bn8 = BatchNormalization()
        self.bn9 = BatchNormalization()
        self.bn10 = BatchNormalization()
        self.bn11 = BatchNormalization()
        self.bn12 = BatchNormalization()
        self.maxpool1d = MaxPooling1D(pool_size=4, padding="same")

        self.dense_classify = Dense(self.output_dim, activation='softmax')

        self.concat = Concatenate()

    # -------------------------------------------#
    #   Attention block
    # -------------------------------------------#
    def attention_3d_block(self,inputs,TIME_STEPS):
        # inputs.shape = (batch_size, time_steps, lstm_units)

        # (batch_size, time_steps, lstm_units) -> (batch_size, lstm_units, time_steps)
        a = Permute((2, 1))(inputs)

        # Dense to last dimention
        # (batch_size, lstm_units, time_steps) -> (batch_size, lstm_units, time_steps)
        a = Dense(TIME_STEPS, activation='softmax')(a)

        a_probs = Permute((2, 1), name='attention_vec')(a)

        output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        return output_attention_mul
    def attention_3d_block2(self,inputs,TIME_STEPS):
        # inputs.shape = (batch_size, time_steps, lstm_units)

        # (batch_size, time_steps, lstm_units) -> (batch_size, lstm_units, time_steps)
        a = Permute((3,2,1))(inputs)

        # (batch_size, lstm_units, time_steps) -> (batch_size, lstm_units, time_steps)
        a = Dense(TIME_STEPS, activation='softmax')(a)

        # (batch_size, lstm_units, time_steps) -> (batch_size, time_steps, lstm_units)
        a_probs = Permute((3,2,1), name='attention_vec2')(a)

        output_attention_mul = merge([inputs, a_probs], name='attention_mul2', mode='mul')
        return output_attention_mul
    def attention_3d_block_spatial(self,inputs,TIME_STEPS):
        # inputs.shape = (batch_size, time_steps, lstm_units)
        # (batch_size, lstm_units, time_steps) -> (batch_size, lstm_units, time_steps)
        a = Conv2D(filters=TIME_STEPS, kernel_size=(1, 1), activation='softmax')(inputs)
        output_attention_mul = merge([inputs, a], name='attention_spatial', mode='mul')
        return output_attention_mul
    def conv_bn_X1(self,inputs,filters):
        x = Conv2D(filters=filters,kernel_size=(1,1))(inputs)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(self.spatial_dropout_rate_1)(x)
        return x
    def Conv_BN_Relu(self,filters, kernel_size, strides, input_layer):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        return x
    def resiidual_a_or_b(self,input_x, filters, flag):
        if flag == 'a':
            # main line
            x = self.Conv_BN_Relu(filters, (3, 3), 1, input_x)
            x = self.Conv_BN_Relu(filters, (3, 3), 1, x)

            # output
            y = Add()([x, input_x])

            return y
        elif flag == 'b':
            # main line
            x = self.Conv_BN_Relu(filters, (3, 3), 2, input_x)
            x = self.Conv_BN_Relu(filters, (3, 3), 1, x)

            # down sample
            input_x = self.Conv_BN_Relu(filters, (1, 1), 2, input_x)

            # output
            y = Add()([x, input_x])

            return y
    def channel_attention(self,input_feature, ratio=8):
        channel_axis = 1 if keras_backend.image_data_format() == "channels_first" else -1
        channel = input_feature._keras_shape[channel_axis]

        shared_layer_one = Dense(channel // ratio,
                                 kernel_initializer='he_normal',
                                 activation='relu',
                                 use_bias=True,
                                 bias_initializer='zeros')

        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        assert avg_pool._keras_shape[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool._keras_shape[1:] == (1, 1, channel)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        assert max_pool._keras_shape[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool._keras_shape[1:] == (1, 1, channel)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('hard_sigmoid')(cbam_feature)

        if keras_backend.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def spatial_attention(self,input_feature):
        kernel_size = 7
        if keras_backend.image_data_format() == "channels_first":
            channel = input_feature._keras_shape[1]
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature._keras_shape[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: keras_backend.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool._keras_shape[-1] == 1
        max_pool = Lambda(lambda x: keras_backend.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool._keras_shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat._keras_shape[-1] == 2
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              activation='hard_sigmoid',
                              strides=1,
                              padding='same',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        assert cbam_feature._keras_shape[-1] == 1

        if keras_backend.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def cbam_block(self,cbam_feature, ratio=8):
        cbam_feature = self.channel_attention(cbam_feature, ratio)
        cbam_feature = self.spatial_attention(cbam_feature, )
        return cbam_feature
    def channel_attention_time(self,input_feature, ratio=8):
        channel_axis = 1 if keras_backend.image_data_format() == "channels_first" else -1
        channel = input_feature._keras_shape[channel_axis]

        shared_layer_one = Dense(channel // ratio,
                                 kernel_initializer='he_normal',
                                 activation='relu',
                                 use_bias=True,
                                 bias_initializer='zeros')

        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling1D()(input_feature)
        avg_pool = Reshape((1, channel))(avg_pool)
        assert avg_pool._keras_shape[1:] == (1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool._keras_shape[1:] == (1, channel // ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool._keras_shape[1:] == (1, channel)

        max_pool = GlobalMaxPooling1D()(input_feature)
        max_pool = Reshape((1, channel))(max_pool)
        assert max_pool._keras_shape[1:] == (1,  channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool._keras_shape[1:] == (1, channel // ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool._keras_shape[1:] == (1, channel)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('hard_sigmoid')(cbam_feature)

        if keras_backend.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])
    def create_model(self):
        input1 = Input(shape=(40, 173, 1), name='mfcc_input')
        input2 = Input(shape=(128, 173), name='raw_input')
        # input1 = self.attention_3d_block_spatial(input1, 64)

        x1 = Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=l2(self.l2_rate), activation='relu')(input1)
        # x1 = self.cbam_block(x1)
        x1 = MaxPooling2D((3, 3), strides=2, padding='same')(x1)
        # x1_X1 = self.conv_bn_X1(x1, 256)

        x2 = self.resiidual_a_or_b(x1, 64, 'b')
        x2 = self.resiidual_a_or_b(x2, 64, 'a')
        # x2 = self.cbam_block(x2)
        x2_x1 = self.conv_bn_X1(x2, 64)
        up2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x1.shape[1], x1.shape[2])))(x2_x1)
        up2_x1 = Add()([x1, up2])
        up2_x1_conv = self.conv_bn_X1(up2_x1, 64)
        # up2_x1_result = Lambda(self.dense_expand)(up2_x1)
        x3 = self.resiidual_a_or_b(up2_x1_conv, 128, 'b')
        x3 = self.resiidual_a_or_b(x3, 128, 'a')
        # x3 = self.cbam_block(x3)
        x3_x1 = self.conv_bn_X1(x3, 64)
        up3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x2.shape[1], x2.shape[2])))(x3_x1)
        up3_x2 = Add()([x2, up3])
        up3_x2_conv = self.conv_bn_X1(up3_x2, 128)
        # up3_x2_result = Lambda(self.dense_expand)(up3_x2)
        x4 = self.resiidual_a_or_b(up3_x2_conv, 64, 'b')
        x4 = self.resiidual_a_or_b(x4, 64, 'a')
        # x4 = self.attention_3d_block_spatial(x4, 64)
        # x4 = self.cbam_block(x4)
        # x4 = self.spatial_attention(x4)
        # x4 = self.channel_attention(x4,8)
        # x4_X1 = self.conv_bn_X1(x4, 128)
        # x4_result = Lambda(self.dense_expand)(x4)
        # x4 = Reshape((110,256))(x4)
        # x4 = Bidirectional(LSTM(256,return_sequences=True))(x4)
        # x4 = Reshape((110,256,1))(x4)
        # l_x_result = self.dense_expand(x4)
        # results = Concatenate()([up2_x1_result,up3_x2_result,x4_result,])
        # up3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x2.shape[1], x2.shape[2])))(x3_X1)
        # up4 =  Lambda(lambda x: tf.image.resize_bilinear(x,size=(x3.shape[1],x3.shape[2])))(x4_X1)
        # x = Add()([x1,up4])
        # x = self.conv_bn(x,64)
        # up4 = x3_X1
        # merge4 = Add()([x3_X1,up4])
        # merge4 = self.conv_bn_X1(merge4,256)

        # up3 = Lambda(lambda x: tf.image.resize_bilinear(x,size=(x2.shape[1],x2.shape[2])))(x3_X1)
        # up3 = x2_X1
        # merge3 = Add()([x2_X1,up3])
        # merge3 = self.conv_bn(merge3,256)
        # up2 = Lambda(self.my_upsampling, arguments={'img_w':x2.shape[1],'img_h':x2.shape[2]})(x2_X1)
        # up_merge4 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x2.shape[1], x2.shape[2])))(merge4)
        # merge = Add()([x2_X1,up_merge4])
        # x = self.attention_3d_block_spatial(x,64)
        x = self.gap(x4)
        # x = self.dense4(x)
        # x = Dropout(0.5)(x)
        # x = self.bn11(x)

        # lstm

        l_x1 = self.lstm1(input2)
        l_x1 = self.dropout1(l_x1)
        l_x1 = self.bn6(l_x1)
        # # # # l_x = self.attention_3d_block_channel(l_x, 64)

        attention_mul = self.channel_attention_time(l_x1, 8)
        # attention_mul = self.attention_3d_block(attention_mul, 128)
        # attention_mul = Flatten()(attention_mul)
        l_x2 = self.lstm2(attention_mul)
        # # # l_x = MaxPooling1D(2)(l_x)
        l_x2 = self.dropout2(l_x2)
        l_x2 = self.bn7(l_x2)
        # l_x2 = Concatenate()([l_x1, l_x2])
        # # # # l_x = Add()([l_x1, l_x2])
        l_x3 = self.lstm3(l_x2)
        l_x3 = self.dropout3(l_x3)
        l_x3 = self.bn8(l_x3)
        # l_x3 = Concatenate()([l_x1,l_x3])
        # # l_x = Concatenate()([l_x1,l_x3])
        # l_x = Reshape((180,512,1))(l_x3)

        # # # l_x = Add()([l_x,l_x3])
        # l_x = GlobalAveragePooling2D()(l_x)
        # # attention_mul = self.attention_3d_block(l_x, 128)
        # # attention_mul = Reshape((128,512,1))(attention_mul)
        l_x = self.dense1(l_x3)
        l_x = self.dropout4(l_x)
        l_x = self.bn9(l_x)
        l_x = self.dense2(l_x)
        l_x = self.dropout5(l_x)
        l_x = self.bn10(l_x)
        # # attention_mul = GlobalAveragePooling2D()(attention_mul)
        l_x = self.dense3(l_x)
        l_x = self.dropout6(l_x)
        l_x = self.bn12(l_x)

        #
        # x_concat = tf.concat([x,l_x],axis = 1)
        # x_concat = merge([x, l_x],mode='mul')
        # x_concat = Concatenate()([x, l_x])
        x_concat = Add()([x,l_x])
        # x_concat = Dense(64)(x_concat)
        # x_concat = Dropout(0.5)(x_concat)
        # x_concat = BatchNormalization()(x_concat)
        # x_concat = Dense(64)(x_concat)
        # x_concat = self.attention_3d_block2(x_concat,128)
        # x_output = Lambda(self.dicision_fusion)(results)
        x_output = self.dense_classify(x_concat)
        # x_output = keras_backend.sum(x_output,axis=1)
        model = Model(inputs=[input1, input2], outputs=[x_output])
        return model

if __name__ == '__main__':
    model = model_mz(5)

    LSTM_CNN = model.create_model()


    mel = np.random.randn(32,60,174,1)
    raw = np.random.randn(32,50,64,20)
    # raw = np.random.randn(32, 1, 88200)
    x = {'mel_input':mel,'raw_input':raw}
    y = LSTM_CNN(x)