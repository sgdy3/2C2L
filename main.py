# -*- coding: utf-8 -*-
# ---
# @File: main.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe: 
# ---


import numpy as np
import tensorflow as tf
from preprocessing import hafemann_preprocess
import keras.utils.np_utils
import re
import pickle
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from model import  TwoC2L


def curve_eval(label,result):
    fpr, tpr, thresholds = roc_curve(label,result, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=result.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size
    area = auc(fpr, tpr)
    print("EER:%f"%EER)
    print('AUC:%f'%area)
    print('ACC(EER_threshold):%f'%acc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()

def load_img(file_name1,file_name2,label,ext_h=820,ext_w=890):
    img1 = tf.io.read_file(file_name1, 'rb')  # 读取图片
    img1 = tf.image.decode_png(img1, channels=3)
    img1 = tf.image.rgb_to_grayscale(img1)
    img1=hafemann_preprocess(img1.numpy(),ext_h,ext_w)
    img1=np.expand_dims(img1,axis=2)

    img2 = tf.io.read_file(file_name2, 'rb')  # 读取图片
    img2 = tf.image.decode_png(img2, channels=3)
    img2 = tf.image.rgb_to_grayscale(img2)
    img2=hafemann_preprocess(img2.numpy(),ext_h,ext_w)
    img2=np.expand_dims(img2,axis=2)

    img=tf.concat([img1,img2],axis=-1)
    return img,label


if __name__=="__main__":
    TC2L=TwoC2L()
    with open('./pair_ind/cedar_ind/train_index.pkl', 'rb') as train_index_file:
        train_ind = pickle.load(train_index_file)
    train_ind = np.array(train_ind)
    train_ind=train_ind[np.random.permutation(train_ind.shape[0]),:]
    dataset = tf.data.Dataset.from_tensor_slices((train_ind[:, 0], train_ind[:, 1], train_ind[:, 2].astype(np.int8)))

    image = dataset.map(
        lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.int8]))
    doc=TC2L.train(image)


    mode='train'

    assert  mode=='train' or mode== 'test', 'the programmer can only execute in training or testing model'

    if mode=='train':

        if(doc): # 进行了训练就画图，直接读档模型不画图
            plt.plot(doc)
            plt.title('categorical_crossentropy')
            plt.xlabel('times')
            plt.ylabel('categorical_crossentropy')

        '''
        给出训练集上所有图片的输出，并进行合并
        '''
        result=[]
        label=[]
        for b in image.batch(32):
            result.append(TC2L.net.predict_on_batch(b))
            label.append(b[1].numpy())

        temp=np.zeros((1,2))
        for i in result:
            temp=np.vstack([temp,i]) # 由于batch为32时不能整除，返回result的shape不都是32不能直接化为ndarray
        temp=temp[1:,:]
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        curve_eval(label,result[:,1])

    else:
        with open('../../pair_ind/cedar_ind/test_index.pkl', 'rb') as test_index_file:
            test_ind = pickle.load(test_index_file)
        test_ind = np.array(test_ind)
        test_set= tf.data.Dataset.from_tensor_slices((test_ind[:, 0], test_ind[:, 1], test_ind[:, 2].astype(np.int8)))
        test_image = test_set.map(
            lambda x, y, z: tf.py_function(func=load_img, inp=[x, y, z], Tout=[tf.uint8,tf.int8]))

        result=[]
        label=[]
        for b in test_image.batch(32):
            result.append(TC2L.net.predict_on_batch(b))
            label.append(b[1].numpy())
        temp=np.zeros((1,2))
        for i in result:
            temp=np.vstack([temp,i]) # 由于batc可能不能整除，返回result的shape不都是batch大小不能直接化为ndarray
        temp=temp[1:,:]
        result=temp.copy()
        temp=np.array([])
        for i in label:
            temp=np.concatenate([temp,i])
        label=temp.copy()
        curve_eval(label,result[:,1])
