import math
import sys
import pickle
import keras
import numpy as np
import os
import theano
import theano.sandbox.cuda.dnn

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD   #for momentum, learning rate
from keras.utils import np_utils

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
     featurewise_center=False,  # set input mean to 0 over the dataset
     samplewise_center=False,  # set each sample mean to 0
     featurewise_std_normalization=False,  # divide inputs by std of the dataset
     samplewise_std_normalization=False,  # divide each input by its std
     zca_whitening=False,  # apply ZCA whitening
     rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
     horizontal_flip=True,  # randomly flip images
     vertical_flip=False)  # randomly flip images





nb_epoch_1 = 300
batch_size = 32
nb_epoch_2 = 200
valid = 50
semi = 1
loadModel = 0
data_augmentation_1 = False
data_augmentation_2 = True

directory_path = sys.argv[1] 
all_label   = pickle.load(open(directory_path + "all_label.p","r"))
if semi :
   all_unlabel = pickle.load(open(directory_path + "all_unlabel.p","r"))
test   = pickle.load(open(directory_path + "test.p","r"))

x_train = np.zeros((len(all_label) * (len(all_label[0])-valid), 3072))
y_train = np.zeros((len(all_label) * (len(all_label[0])-valid), 10))
x_train = np.reshape(x_train,(len(all_label) * (len(all_label[0])-valid),3,32,32))

x_valid = np.zeros((len(all_label) * valid, 3072))
y_valid = np.zeros((len(all_label) * valid, 10))
x_valid = np.reshape(x_valid,(len(all_label) * valid,3,32,32))

if semi == 1:
   x_unlabel = np.zeros((len(all_unlabel), 3072))
   x_unlabel = np.reshape(x_unlabel,(len(all_unlabel),3,32,32))

test_data = np.zeros((10000,3072))
test_data = np.reshape(test_data,(10000,3,32,32))

cnt = 0
cnt_2 = 0

for i in range(len(all_label)):
   for j in range(len(all_label[0])):
       for n in range(3):
          for k in range(32):
              for m in range(32):
                 if j >= len(all_label[0]) - valid:
                     x_valid[cnt_2][n][k][m] = all_label[i][j][n * 1024 + k * 32 + m]
                 else :
                     x_train[cnt][n][k][m] = all_label[i][j][n * 1024 + k * 32 + m]
       if j >= len(all_label[0]) - valid:
           y_valid[cnt_2][i] = 1
           cnt_2 += 1
       else :
           y_train[cnt][i] = 1
           cnt += 1
if semi == 1:
    for i in range(len(all_unlabel)):
        for n in range(3):
           for k in range(32):
               for m in range(32):
                   x_unlabel[i][n][k][m] = all_unlabel[i][n * 1024 + k * 32 + m]

print "x_train.shape :", x_train.shape
print "y_train.shape :", y_train.shape
print "x_valid.shape :", x_valid.shape
print "y_valid.shape :", y_valid.shape
if semi == 1: 
    print "x_unlabel.shape :", x_unlabel.shape


for i in range(10000):
    for n in range(3):
       for k in range(32):
           for m in range(32):
              test_data[i][n][k][m] = test['data'][i][n * 1024 + k * 32 + m]

test_data = test_data.astype('float32')
test_data /= 255

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
if semi == 1: x_unlabel = x_unlabel.astype('float32')
x_train /= 255
x_valid /= 255
if semi == 1: x_unlabel /= 255


img_rows, img_cols = 32, 32
img_channels = 3

if loadModel == 1:
    model = load_model(str(sys.argv[5]))
else :
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = (img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25)) 

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  
    model.add(Dense(10))
    keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
    model.add(Activation('softmax'))

    keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])

    if not data_augmentation_1:
        print('Not using data augmentation.')
        for epoch in xrange(nb_epoch_1):
            print "epoch = ", epoch
            model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = 1)
            score = model.evaluate(x_valid, y_valid, batch_size = len(x_valid))
            
            print ('Test score/Top1', score)

    else:
        print('Using real-time data augmentation.')
 
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)
 
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=nb_epoch_1,
                        validation_data=(x_valid, y_valid))

##############

for n in range(1):
    if semi == 1:
        p = model.predict_proba(x_unlabel, batch_size = 100, verbose = 0) 
        add = 0
        add_x_train = [ ]
        add_y_train = [ ]
        delete_id = [ ]
        for i in range(len(p)):
            ans = 0
            id = 0
            for j in range(10):
                if p[i][j] > ans: 
                    ans = p[i][j]
                    ans_id = j    
            for j in range(10):
                if ans_id == j: p[i][j] = 1
                else : p[i][j] = 0
            if ans > 0.9999 and add < 3000:
                add += 1
                add_x_train.append(x_unlabel[i])
                add_y_train.append(p[i])
                delete_id.append(i)
        for i in range(len(delete_id)):
            x_unlabel = np.delete(x_unlabel, (i), axis = 0)
        add_x_train = np.array(add_x_train)
        add_y_train = np.array(add_y_train)
 
        x_train = np.concatenate(( x_train, add_x_train))
        y_train = np.concatenate(( y_train, add_y_train))
        
        if not data_augmentation_2:
            print('Not using data augmentation.')
            for epoch in xrange(nb_epoch_2):
                print "epoch = ", epoch
                model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = 1)
                score = model.evaluate(x_valid, y_valid, batch_size = len(x_valid))
                
                print ('Test score/Top1', score)
 
        else:
            print('Using real-time data augmentation.')
  
            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(x_train)
  
            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(x_train, y_train,
                            batch_size=batch_size),
                            samples_per_epoch=x_train.shape[0],
                            nb_epoch=nb_epoch_2,
                            validation_data=(x_valid, y_valid))
    

model.save(sys.argv[2])

