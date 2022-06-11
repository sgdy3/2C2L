# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/22
# Describe: 实现了Siamese网络进行脱机签名认证
# ---


import keras.utils.np_utils
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Dense,GlobalAveragePooling2D,Subtract,Softmax
from keras.models import Sequential,Model
from keras.layers import Activation,Dropout,Input,Flatten
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
import keras
import os
import numpy as np

class TwoC2L():
    def __init__(self):
        self.rows=150
        self.cols=220
        self.channles=2
        self.imgshape = (self.rows, self.cols, self.channles)

        self.batchsize=64
        self.epochs=1


        self.subnet=self.bulid_model()
        self.optimizer= adam_v2.Adam(learning_rate=0.0003)

        sig=Input(shape=self.imgshape)
        label=Input(shape=(1,))

        feature=self.subnet(sig)
        logit1=Dense(2,activation='relu',name='logit1')(feature)
        logit2=Dense(2,activation='relu',name='logit2')(feature)
        subtracted=Subtract(name='sub')([logit1,logit2])
        out=Softmax()(subtracted)

        self.net=Model([sig,label], out)
        self.net.compile(loss='categorical_crossentropy',metrics='accuracy',optimizer=self.optimizer)
        plot_model(self.net, to_file='2-channle.png', show_shapes=True)
        self.net.summary()


    def bulid_model(self):
        model=Sequential()

        model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv1',activation='relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding='same',input_shape=(self.rows,self.cols,self.channles),name='conv2',activation='relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='pool1'))

        model.add(Conv2D(64, kernel_size=(5, 5), strides=1, padding='same',name='conv3',activation='relu'))
        model.add(BatchNormalization(momentum=0.9))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool2'))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',name='conv4',activation='relu'))

        model.add(Conv2D(96, kernel_size=(3, 3), strides=1, padding='same',name='conv5',activation='relu'))

        model.add(MaxPool2D(pool_size=(3, 3), strides=2,name='pool3'))
        model.add(Dropout(0.3))

        model.add(GlobalAveragePooling2D())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(256,name='fc1'))
        model.add(Dense(128,name='fc2',activation='relu'))

        model.summary()

        img=Input(shape=self.imgshape)
        feature=model(img)
        return  Model(img,feature,name='subnet')


    def train(self,dataset,weights='',save=False):
        save_dir = '../../NetWeights/2C2L_weights'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if weights:
            filepath = os.path.join(save_dir, weights)
            self.net.load_weights(filepath)
        else:
            filepath = os.path.join(save_dir, '2C2L.h5')
            dataset = dataset.shuffle(100).batch(self.batchsize).repeat(self.epochs)
            i=1
            doc=[]
            pre_loss=[]
            min_loss=100 # manually set
            for batch in dataset:
                train_label=keras.utils.np_utils.to_categorical(batch[1])
                loss,acc = self.net.train_on_batch(batch,y=train_label)  # batch[1] are labels
                doc.append(loss)
                print("round %d=> loss:%f, acc:%f%% " % (i,loss,acc*100))
                if(early_stop(20,loss,pre_loss,threshold=0.005)):
                    print("training complete")
                    break
                if(i>500):
                    print("enough rounds!!")
                    break
                i+=1
            if save:
                self.net.save_weights(filepath)
            return doc

def early_stop(stop_round,loss,pre_loss,threshold=0.005):
    '''
    early stop setting
    :param stop_round: rounds under caculated
    :param pre_loss: loss list
    :param threshold: minimum one-order value of loss list
    :return: whether or not to jump out
    '''
    if(len(pre_loss)<stop_round):
        pre_loss.append(loss)
        return False
    else:
        loss_diff=np.diff(pre_loss,1)
        pre_loss.pop(0)
        pre_loss.append(loss)
        if(abs(loss_diff).mean()<threshold): # to low variance means flatten field
            return True
        else:
            return False