#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import os
import numpy as np
from sklearn.model_selection import ShuffleSplit
import SimpleITK as sitk
import matplotlib.pyplot as plt
from keras import backend as K

#function for creating the path to image
def create_px_path(px_list):
    img_paths = []
    seg_paths = []
    px_fol = '/content/UNET/try'

    for number in px_list:
        img_path = glob.glob(os.path.join(px_fol, f'{number}', 'images', '*.nii'))
        seg_path = glob.glob(os.path.join(px_fol, f'{number}', 'segmentations', '*.nii'))
        img_paths.extend(img_path)
        seg_paths.extend(seg_path)

    return img_paths, seg_paths

#function to create a normalized image 
def percentile_norm(datas):
    x_95 = np.percentile(datas, 95)
    x_05 = np.percentile(datas, 5)
    datas = datas.astype(np.float64)  # Convert to floating-point type
    datas -= x_05
    datas /= (x_95 - x_05)
    return datas

def split_data_train_val_test(px_fol):

    patients = os.listdir(px_fol)
    patients = np.asarray(patients)

    ss    = ShuffleSplit(n_splits=1,test_size=0.20)
    ss.get_n_splits(patients)

    for train_index, test_index in ss.split(patients):
        xt, x_test = patients[train_index], patients[test_index]

    ss = ShuffleSplit(n_splits=1, test_size=0.20)
    ss.get_n_splits(xt)

    for ten_index, val_index in ss.split(xt):
        x_train_in, x_val_in= xt[ten_index], xt[val_index]

    px_splits = {'train': np.ndarray.tolist(x_train_in),
                 'val'  : np.ndarray.tolist(x_val_in)  ,
                 'test' : np.ndarray.tolist(x_test)    }

    return px_splits

#batch generator 
def generate_batch_norm(batch): #for images
    data = []

    for img in batch:
        #print('img = '+str(img))
        img_data = sitk.ReadImage(img)

        img_data = sitk.GetArrayFromImage(img_data)
        img_data = img_data.astype('float32')
        img_data = percentile_norm(img_data)
        data.append(img_data)

    data = np.stack(data)


    data = np.reshape(data, (data.shape[0],data.shape[2],data.shape[1],1))
    return data
def generate_batch(batch): #for labels image
    data = []

    for img in batch:
        #print('img = '+str(img))
        img_data = sitk.ReadImage(img)

        img_data = sitk.GetArrayFromImage(img_data)
        img_data = img_data.astype('float32')
        data.append(img_data)

    data = np.stack(data)


    data = np.reshape(data, (data.shape[0],data.shape[2],data.shape[1],1))
    return data

# data generator 
def data_generator(x_lis,y_tar,d_size):
    while True:
        len_lis = len(x_lis)
        nu_part = (len_lis//d_size)+1
        count   = 0

        for i in range(nu_part):

            if count >= len_lis:
                continue

            if i+1 == d_size:
               p_list = x_lis[count:]
               segment= y_tar[count:]
            else:
               p_list  = x_lis[count:count+d_size]
               segment = y_tar[count:count+d_size]

            images = generate_batch_norm(p_list)
            target = generate_batch(segment)
            count += d_size
            yield images, target
            
#dice 
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

