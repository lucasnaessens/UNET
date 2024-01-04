#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install Keras-Preprocessing')
from keras import Input
from keras.models import Model
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.layers import Concatenate, Conv2D
from keras_preprocessing.image import array_to_img
from keras.callbacks import ModelCheckpoint
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



from keras.utils import plot_model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    K.print_tensor(intersection, message="Dice intersection:")
    return -((2. * intersection + K.epsilon()) / (K.sum(y_true_f)
                                                  + K.sum(y_pred_f)
                                                  + K.epsilon()))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    K.print_tensor(intersection, message="Dice intersection:")
    return -((2. * intersection + K.epsilon()) / (K.sum(y_true_f)
                                                  + K.sum(y_pred_f)
                                                  + K.epsilon()))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_unet():
    '''
    Create a U-Net
    '''
    print('Creating U-Net...')

    # First, we have to provide the dimensions of the input images
    inputs = Input((528, 1200, 1))

    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    print('conv1 shape:', conv1.shape)
    print('pool1 shape:', pool1.shape)

    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    print('conv2 shape:', conv2.shape)
    print('pool2 shape:', pool2.shape)


    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    drop4 = Dropout(0.5)(conv3)  # Added
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    print('conv3 shape:', conv3.shape)
    print('pool3 shape:', pool3.shape)

    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    print('conv4 shape:', conv4.shape)


    up7 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(drop4))  # Changed
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(32, 3,activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)
    print('conv7 shape:', conv7.shape)

    up8 = Conv2D(16, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)
    print('conv8 shape:', conv8.shape)

    up9 = Conv2D(2, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(2, 3,activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    print('conv9 shape:', conv9.shape)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    print('conv10 shape:', conv10.shape)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer='adam',
                  loss=dice_coef, metrics=[dice_coef_loss])



    print('Got U-Net!')

    return model

