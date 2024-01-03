#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from functions import *
from model import *

import glob
import os
import numpy as np
from sklearn.model_selection import ShuffleSplit
import SimpleITK as sitk
import matplotlib.pyplot as plt
from keras import backend as K

#downloading data 
get_ipython().system(' git clone https://github.com/lucasnaessens/UNET.git')
#creating dataset

px_fol_path = '/content/UNET/try/'
px_split_di = split_data_train_val_test(px_fol_path)
x_train , y_train = create_px_path(px_split_di['train'])
x_valid , y_valid = create_px_path(px_split_di['val'])
x_test  , y_test  = create_px_path(px_split_di['test'])

#importing model 
model = create_unet()

#training 
epch = 250
batch_size = 5


filepath='/content/UNET/weights_new/w_epoch_.{epoch:02d}_'+'.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef_loss',period=5, verbose=1, save_best_only=True,
                              mode='max')

callbacks_list = [checkpoint]

#train with real time augmentation 
train_generator = datagen.flow(generate_batch_norm(x_train), generate_batch(y_train), batch_size=batch_size)


validation_generator = datagen.flow(generate_batch_norm(x_valid), generate_batch(y_valid), batch_size=batch_size)

# Train the model using the flow method
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epch,
    validation_data=validation_generator,
    validation_steps=len(x_valid) // batch_size,
    callbacks=callbacks_list
)

