import keras
from keras.models import Model, model_from_json
from keras.layers import Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, merge, Dropout, ZeroPadding2D, Lambda
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam, Adagrad
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import CSVLogger

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse

from prepare_ecs import *
from models import *

import sys
import numpy as np
import random
import time
import scipy.misc as misc
from datetime import datetime
from time import gmtime, strftime

models_dir = '/n/coxfs01/eric_wu/convlstm/models/'
checkpoints_dir = '/n/coxfs01/eric_wu/convlstm/checkpoints/'
progress_dir = '/n/home06/ericwu/convlstm/progress/'
numKernels = 32
weight_decay = 0.0
momentum = 0.99
learn_rate = 0.00001

num_iterations = 200000
steps_per_epoch = 100

initialization = 'glorot_uniform'

params = {
        'numKernels':numKernels,
        'weight_decay':weight_decay,
        'initialization':initialization
        }

mb = ModelBuilder(*params)

newModel = True
filename = "runet"
modelType = filename
doBinary = False
doEdt = False

#modelClass = getModelClassFromFilename(modelType)
modelClass = getModelClassFromFilename(filename)
#modelClass.setBaseFilename(models_dir+"runet_graft")
filename = "runet_halfres"
(patchZ, patchZ_out, patchSize, patchSize_out) = modelClass.getPatchSizes()
loss = modelClass.getLoss()
optimizer = "adam"

print "New model: ", newModel
print "Model type:", modelType
print "Filename:", filename
print "Do binary:",doBinary
print "Do edt:", doEdt
print "Patch sizes:", modelClass.getPatchSizes()
print "Num kernels:", numKernels
print "Learning rate:", learn_rate
print "num_iterations", num_iterations
print "steps_per_epoch", steps_per_epoch
print "loss function", loss
print "optimizer", optimizer

cropSize = (patchSize - patchSize_out)/2
csZ = (patchZ - patchZ_out)/2

def train(model, generator, callbacks=None, validator=None):
    if validator is not None:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=1, max_q_size=5,
                            validation_data=validator, validation_steps=25, callbacks=callbacks)
    else:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=1, max_q_size=5, callbacks=callbacks)
    print "One epoch trained!"

def save(model):
    open(models_dir + filename + '.json', 'w').write(model.to_json())
    model.save_weights(models_dir + filename + '_weights.h5', overwrite=True)

def save_checkpoint(model, i):
    num = datetime.now().strftime("%Y-%m-%d-%H:%M")
    model.save_weights(checkpoints_dir+filename+'_'+str(i)+'_'+str(num)+'_weights.h5', overwrite=True)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        np.savetxt(progress_dir+filename+".txt", np.array(self.losses), delimiter=",")


if (newModel):
    print "Building model", filename
    model = modelClass.build()
    open(models_dir + filename + '.json', 'w').write(model.to_json())
    model.save_weights(models_dir + filename + '_weights.h5', overwrite=True)
else:
    print "Loading model", filename
    model = model_from_json(open(models_dir + filename + '.json').read())
    model.load_weights(models_dir + filename + '_weights.h5')

print "Compiling model", filename
csv_logger = CSVLogger(progress_dir+filename+".csv", append=True, separator=';')
lr_schedule = lambda epoch: learn_rate if epoch < 100 else learn_rate*0.1    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [csv_logger]

if optimizer == "adam":
    opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
else:
    opt = SGD(lr=learn_rate, decay=0, momentum=momentum, nesterov=False)
#loss = 'binary_crossentropy'
model.compile(loss=loss, optimizer=opt)

print "Generating dataset"

data_train = generate_dataset(dataset="ecs_train_halfres", cropSize=cropSize, csZ=csZ, doEdt=doEdt)
generator_train = generate_samples(nsamples=1, doAugmentation=True, patchSize=patchSize,
                                    patchSize_out=patchSize_out, patchZ=patchZ, patchZ_out=patchZ_out,
                                    doBinary=doBinary, modelType=modelType, dataset=data_train)
'''
if doBinary:
    data_valid = generate_dataset(dataset="a", cropSize=cropSize, csZ=csZ, split="val")
    generator_valid = generate_samples(nsamples=1, doAugmentation=True, patchSize=patchSize,
                                        patchSize_out=patchSize_out, patchZ=patchZ, patchZ_out=patchZ_out,
                                        doBinary=doBinary, modelType=modelType, dataset=data_valid)
'''
i = 0
prev_time = int(time.time())
start_time = prev_time
while i < num_iterations/steps_per_epoch:
    print "Training model for iteration", i
    cur_time = int(time.time())
    print "Current timestamp:", cur_time
    print "Seconds elapsed since last batch:", cur_time - prev_time
    print "Seconds elapsed since start of training:", cur_time - start_time
    prev_time = cur_time
    train(model, generator_train, callbacks)
    '''
    if doBinary:
        train(model, generator_train, validator=generator_valid)
    else:
        train(model, generator_train)
    '''

    i += 1
    if i % 3 == 0:
        print "Saving model for iteration", i
        save(model)
    if i % 10 ==0:
        save_checkpoint(model, i)