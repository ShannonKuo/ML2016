import math
import sys
import pickle
import keras
import numpy as np
import os
import theano
#os.environ["THEANO_FLAGS"] = "mode=FAST,device=gpu,floatX=float32"

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD   #for momentum, learning rate
from keras.utils import np_utils

model = load_model(sys.argv[2])

directory_path = sys.argv[1] 
test   = pickle.load(open(directory_path + "test.p","r"))

test_data = np.zeros((10000,3072))
test_data = np.reshape(test_data,(10000,3,32,32))


for i in range(10000):
    for n in range(3):
       for k in range(32):
           for m in range(32):
              test_data[i][n][k][m] = test['data'][i][n * 1024 + k * 32 + m]

test_data = test_data.astype('float32')
test_data /= 255
predict = open(sys.argv[3], "w")
p = model.predict(test_data, batch_size = 100, verbose = 0) 

predict.write("ID,class\n")
for i in range(len(p)):
    ans = 0
    id = 0
    for j in range(10):
        if p[i][j] > ans: 
            ans = p[i][j]
            ans_id = j    
    predict.write(str(i)+","+str(ans_id)+"\n")
"""
for i in range(len(10000)):
    test_ID[i][0] = test[i]



for i in range(len(10000)):
   for j in range(len(all_label[0])):
       for n in range(3):
          for k in range(32):
              for m in range(32):
                 x_train[cnt][k][m][n] = all_label[i][j][n * 1024 + k * 32 + m]
       y_train[cnt][i] = 1
       cnt += 1
"""
