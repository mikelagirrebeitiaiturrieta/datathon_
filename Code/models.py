# -*- coding: utf-8 -*-
"""
//////////////////////////////////////////////////////////////////////////////////////////
// Original author: Aritz Lizoain
// Github: https://github.com/aritzLizoain
// My personal website: https://aritzlizoain.github.io/
// Description: CNN Image Segmentation
// Copyright 2020, Aritz Lizoain.
// License: MIT License
//////////////////////////////////////////////////////////////////////////////////////////

ARCHITECTURE: U-Net
Original: https://arxiv.org/pdf/1505.04597.pdf
"""
import numpy as np 
import os
import matplotlib.pyplot as plt   

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.losses
from keras import regularizers #fixing overfitting with L2 regularization
import tensorflow.keras.backend as K
import tensorflow as tf
from typing import Callable

import keras
from PIL import Image
import io
import numpy as np
import tensorflow as t


#############################################################################

#Imbalanced dataset --> weighted loss function cross entropy is needed
#Images too biased towards the first class (background ~95%)

#WEIGHTED LOSS FUNCTION CROSS ENTROPY
def weighted_categorical_crossentropy(weights= [1.,1.,1.,1.]):
    print('The used loss function is: weighted categorical crossentropy')
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

def convert_to_logits(y_pred: tf.Tensor) -> tf.Tensor:
    """
    Converting output of sigmoid to logits.
    :param y_pred: Predictions after sigmoid (<BATCH_SIZE>, shape=(None, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1)).
    :return: Logits (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
    """
    # To avoid unwanted behaviour of log operation
    y_pred = K.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return K.log(y_pred / (1 - y_pred))

def binary_weighted_cross_entropy(beta: float, is_logits: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted cross entropy. All positive examples get weighted by the coefficient beta:
        WCE(p, p̂) = −[β*p*log(p̂) + (1−p)*log(1−p̂)]
    To decrease the number of false negatives, set β>1. To decrease the number of false positives, set β<1.
    If last layer of network is a sigmoid function, y_pred needs to be reversed into logits before computing the
    weighted cross entropy. To do this, we're using the same method as implemented in Keras binary_crossentropy:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    Used as loss function for binary image segmentation with one-hot encoded masks.
    :param beta: Weight coefficient (float)
    :param is_logits: If y_pred are logits (bool, default=False)
    :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the weighted cross entropy.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        if not is_logits:
            y_pred = convert_to_logits(y_pred)

        wce_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(wce_loss))
        wce_loss = K.mean(wce_loss, axis=axis_to_reduce)

        return wce_loss

    return loss

#----------------------------------------------------------------------------

"""
UNET

It takes images (n_img, h, w, 3(rgb)) and masks (n_img, h, w, n_classes) for training
Output has shape (n_img, h, w, n_classes)

Comments are prepared to change number of layers
"""

def unet(pretrained_weights = None, input_size = (256,256,3), weights= [1.,1.,1.,1.],\
         activation='relu', dropout=0, loss='categorical_crossentropy', optimizer='adam',\
             dilation_rate=(1,1), reg=0.01):
    
    inputs = Input(input_size)
    s = Lambda(lambda x: x / 255) (inputs)
    
    #CONTRACTIVE Path (ENCODER)    
    
    # cm3 = Conv2D(2, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (s)
    # cm3 = Dropout(dropout) (cm2)
    # cm3 = Conv2D(2, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (cm2)
    # pm3 = MaxPooling2D((2, 2)) (cm2)
    # pm3 = BatchNormalization()(pm2)
    
    # cm2 = Conv2D(4, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (s)
    # cm2 = Dropout(dropout) (cm2)
    # cm2 = Conv2D(4, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (cm2)
    # pm2 = MaxPooling2D((2, 2)) (cm2)
    # pm2 = BatchNormalization()(pm2)

    # cm1 = Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (s)
    # cm1 = Dropout(dropout) (cm1)
    # cm1 = Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (cm1)
    # pm1 = MaxPooling2D((2, 2)) (cm1)
    # pm1 = BatchNormalization()(pm1)
    
    # c0 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (pm1)
    # c0 = Dropout(dropout) (c0)
    # c0 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (c0)
    # p0 = MaxPooling2D((2, 2)) (c0)
    # p0 = BatchNormalization()(p0)

    c1 = Conv2D(32, 3 , activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (inputs)
    c1 = Dropout(dropout) (c1)
    c1 = Conv2D(32, 3 , activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    # p1 = BatchNormalization()(p1)

    c2 = Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p1)
    c2 = Dropout(dropout) (c2)
    c2 = Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    # p2 = BatchNormalization()(p2)

    c3 = Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p2)
    c3 = Dropout(dropout) (c3)
    c3 = Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    # p3 = BatchNormalization()(p3)

    c4 = Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p3)
    c4 = Dropout(dropout) (c4)
    c4 = Conv2D(256, 3, activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    # p4 = BatchNormalization()(p4)

    c5 = Conv2D(512, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (p4)
    c5 = Dropout(dropout) (c5)
    c5 = Conv2D(512, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c5)


    #EXPANSIVE Path (DECODER)
    
    u6 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u6)
    c6 = Dropout(dropout) (c6)
    c6 = Conv2D(256, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c6)
    
    u7 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u7)
    c7 = Dropout(dropout) (c7)
    c7 = Conv2D(128, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c7)
    
    u8 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u8)
    c8 = Dropout(dropout) (c8)
    c8 = Conv2D(64, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c8)

    u9 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (u9)
    c9 = Dropout(dropout) (c9)
    c9 = Conv2D(32, 3, activation=activation, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(reg)) (c9)
    
    # u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
    # u10 = concatenate([u10, c0], axis=3)
    # c10 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (u10)
    # c10 = Dropout(dropout) (c10)
    # c10 = Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (c10)
    
    # u11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c10)
    # u11 = concatenate([u11, cm1], axis=3)
    # c11 = Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (u11)
    # c11 = Dropout(dropout) (c11)
    # c11 = Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (c11)

    # u12 = Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same') (c10)
    # u12 = concatenate([u12, cm2], axis=3)
    # c12 = Conv2D(4, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (u12)
    # c12 = Dropout(dropout) (c12)
    # c12 = Conv2D(4, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (c12)

    # u13 = Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same') (c11)
    # u13 = concatenate([u12, cm2], axis=3)
    # c13 = Conv2D(2, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (u12)
    # c13 = Dropout(dropout) (c12)
    # c13 = Conv2D(2, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same', kernel_regularizer=regularizers.l2(0.01)) (c12)
 
    #softmax as activaition in the last layer
    outputs = Conv2D(1, 1, activation='softmax') (c9) 

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss,\
                  metrics = ['accuracy'])
    #model.summary() 
    
    if(pretrained_weights):
        
        print('Using {0} pretrained weights'.format(pretrained_weights))
    
        model.load_weights(pretrained_weights)

    return model



class ImageHistory(keras.callbacks.Callback):
    def __init__(self, data, draw_interval=1, log_dir='./logs'):
        super().__init__()
        self.data = data
        self.draw_interval = draw_interval
        self.log_dir = log_dir
        self.last_step = 0

    def on_batch_end(self, batch, logs={}):
        if batch % self.draw_interval == 0:
            images = []
            labels = []
            for item in self.data:
                image_data = item[0]
                label_data = item[1]
                y_pred = self.model.predict(image_data)
                images.append(y_pred)
                labels.append(label_data)
            image_data = np.concatenate(images,axis=2)
            label_data = np.concatenate(labels,axis=2)
            data = np.concatenate((image_data,label_data), axis=1)
            self.last_step += 1
            self.saveToTensorBoard(data, 'batch',
                self.last_step*self.draw_interval)
        return

    def make_image(self, npyfile):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from 
        https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = npyfile.shape
        image = Image.frombytes('L',(width,height),
                                npyfile.tobytes())
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.compat.v1.Summary.Image(height=height,
                                width=width, colorspace=channel,
                                encoded_image_string=image_string)

    def saveToTensorBoard(self, npyfile, tag, epoch):
        data = npyfile[0,:,:,:]
        image = (((data - data.min()) * 255) / 
                (data.max() - data.min())).astype(np.uint8)
        image = self.make_image(image)
        summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag=tag,
                    image=image)])
        writer = tf.compat.v1.summary.FileWriter(
                    self.tensor_board_dir)
        writer.add_summary(summary, epoch)
        writer.close()

