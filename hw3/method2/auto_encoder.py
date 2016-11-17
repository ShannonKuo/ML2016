import math
import sys
import pickle
import keras
import numpy as np
import os
import theano
import theano.sandbox.cuda.dnn
import pydot
from sklearn.manifold import TSNE

#os.environ["THEANO_FLAGS"] = "device = gpu0" 
#print theano.sandbox.cuda.dnn.version()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD, RMSprop  #for momentum, learning rate
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils.visualize_util import plot

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




nb_epoch = 200
batch_size = 32
valid = 50
length = 512
add_size = 1000
data_augmentation = True

directory_path = sys.argv[1] 
all_label   = pickle.load(open(directory_path + "all_label.p","r"))
all_unlabel = pickle.load(open(directory_path + "all_unlabel.p","r"))
test   = pickle.load(open(directory_path + "test.p","r"))

cnt = 0
cnt_2 = 0

x_train = np.zeros((len(all_label) * (len(all_label[0])-valid), 3072))
y_train = np.zeros((len(all_label) * (len(all_label[0])-valid), 10))
x_unlabel = np.zeros((len(all_unlabel), 3072))

x_valid = np.zeros((len(all_label) * valid, 3072))
y_valid = np.zeros((len(all_label) * valid, 10))

test_data = np.zeros((10000,3072))
test_data = np.reshape(test_data,(10000,3,32,32))


for i in range(len(all_label)):
   for j in range(len(all_label[0])):
       for n in range(3072):
           if j >= len(all_label[0]) - valid:
               x_valid[cnt_2][n] = all_label[i][j][n]
           else :
               x_train[cnt][n] = all_label[i][j][n]
       if j >= len(all_label[0]) - valid:
           y_valid[cnt_2][i] = 1
           cnt_2 += 1
       else :
           y_train[cnt][i] = 1
           cnt += 1
x_unlabel = np.reshape(all_unlabel, (len(all_unlabel), 3072))
print "x_train.shape :", x_train.shape
print "y_train.shape :", y_train.shape
print "x_valid.shape :", x_valid.shape
print "y_valid.shape :", y_valid.shape


for i in range(10000):
    for n in range(3):
       for k in range(32):
           for m in range(32):
              test_data[i][n][k][m] = test['data'][i][n * 1024 + k * 32 + m]

model = Sequential()

model.add( Dense( length, activation = 'relu', input_shape = ( 3 * 32 * 32, ) ) )
model.add( Dense( length, activation = 'relu' ) )
model.add( Dense( length, activation = 'relu' ) )
model.add( Dense( length, activation = 'relu' ) )
model.add( Dense( 3 * 32 * 32, activation = 'linear' ) )

keras.optimizers.SGD(lr = 1, decay = 1e-6, momentum = 0.9, nesterov = True) #?nesterov
model.compile(loss = 'mse', optimizer = RMSprop(), metrics = ['accuracy']) #?accuracy


x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_unlabel = x_unlabel.astype('float32')
x_train /= 255
x_valid /= 255
x_unlabel /= 255
test_data = test_data.astype('float32')
test_data /= 255



model.fit(x_train, x_train, nb_epoch = nb_epoch, batch_size = batch_size, shuffle=True)


score = model.evaluate(x_valid, x_valid, batch_size = len(x_valid))
    
print ('Test score/Top1', score)
encode = K.function([model.layers[0].input],
                    [model.layers[3].output]) 
encode_output = encode([x_train])[0]

class_mean = np.zeros((10, length))

for i in range(4500):
    for j in range(length):
        class_mean[i / 450][j] += encode_output[i][j]
for i in range (10):
    for j in range(length):
        class_mean[i][j] /= 450.0
encode_output = encode([x_unlabel])[0]

add = [ ]
y_ans = [0 for i in range(10)]
for i in range(45000):
    diff = [0 for j in range(10) ]
    min_diff = -1
    for j in range(10):
        for k in range(length):
            diff[j] += abs(encode_output[i][k] - class_mean[j][k])   
    for j in range(10):
        if diff[j] < min_diff or min_diff == -1:
            min_diff = diff[j]
            ans = j
    add.append((i, ans, min_diff))

add.sort(key = lambda x : -x[2])

add_x_train = [ ]
add_y_train = [ ]
for i in range( add_size ) :
    tmp_y = [ 0.0 for _ in range( 10 ) ]
    tmp_y[ add[ i ][ 1 ] ] = 1.0
    add_x_train.append( all_unlabel[ add[ i ][ 0 ] ] )
    add_y_train.append( tmp_y )
add_x_train = np.array(add_x_train)
add_y_train = np.array(add_y_train)

add_x_train = add_x_train.astype( 'float32' )
add_y_train = add_y_train.astype( 'float32' )
add_x_train /= 255

x_train = np.concatenate( (x_train, add_x_train) )
y_train = np.concatenate( (y_train, add_y_train) )

x_train = x_train.reshape((5000 - valid * 10 + add_size, 3, 32, 32))
x_valid = x_valid.reshape((valid * 10, 3, 32, 32))

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = (3, 32, 32)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(64, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))


model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(512)) 
model.add(Activation('relu'))
model.add(Dropout(0.25)) 
model.add(Dense(10))
keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
model.add(Activation('softmax'))

keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy']) 

if not data_augmentation:
    print('Not using data augmentation.')
    for epoch in xrange(nb_epoch):
        print "epoch = ", epoch
        model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = 1)
        score = model.evaluate(x_valid, y_valid, batch_size = len(x_valid))
        
        print ('Test score/Top1', score)

else:
    print('Using real-time data augmentation.')
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(x_valid, y_valid))


    print ( x_train.shape )
    print ( x_train.shape[ 0 ] )
    print ( y_train.shape )
    print ( y_train.shape[ 0 ] )

model.save(sys.argv[2])

