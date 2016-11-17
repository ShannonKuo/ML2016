#!/bin/bash

#THEANO_FLAGS=mode=FAST_RUN,,device=gpu,floatX=float32 python selfTraining.py $1 $2
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1 python selfTraining.py $1 $2
python auto_encoder.py $1 $2
