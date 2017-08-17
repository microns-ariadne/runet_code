import keras
from keras.models import Model, model_from_json
from keras.layers import Activation, Flatten, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, merge, Dropout, ZeroPadding2D, Lambda
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, merge, Dropout, Lambda
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, Adagrad
from keras.regularizers import l2
from keras import backend as K
from keras.applications.vgg16 import VGG16

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse

import sys
import numpy as np
import random
import time
import scipy.misc as misc
from datetime import datetime

class ModelBuilder(object):

    def __init__(self, numKernels=32, weight_decay=0.0, initialization='glorot_uniform'):
        self.numKernels = numKernels
        self.weight_decay = weight_decay
        self.initialization = initialization
        self.patchZ = 0
        self.patchZ_out = 0
        self.patchSize = 0
        self.patchSize_out = 0
        self.filename = ""
        self.loss = ""

    def getPatchSizes(self):
        return (self.patchZ, self.patchZ_out, self.patchSize, self.patchSize_out)

    def setPatchSizes(self, patches):
        self.patchZ = patches[0]
        self.patchZ_out = patches[1]
        self.patchSize = patches[2]
        self.patchSize_out = patches[3]

    def getFilename(self):
        return self.filename

    def setFilename(self, fn):
        self.filename = fn

    def getLoss(self):
        return self.loss

class Unet2D(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 1
        self.patchZ_out = 1
        self.patchSize = 508
        self.patchSize_out = 324
        self.filename = "unet_2d"
        self.loss = weighted_mse

    def build(self):

        def retanh(x):
            return K.maximum(0, K.tanh(x))
        
        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(input)

            act2 = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)

            if doPooling:
                pool1 = MaxPooling2D(pool_size=(2, 2))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = UpSampling2D(size=(2,2))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = Conv2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu')(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
            
            print "cropped layer size: ", down_block_out_cropped._keras_shape
            merged = concatenate([conv_up, down_block_out_cropped], axis=1)

            print "merged ", merged._keras_shape
            act1 = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu')(merged)

            print "conv1 ", act1._keras_shape
            act2 = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_act._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_act._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_act._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_act._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_act._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="
        
        conv = Conv2D(filters=3, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh)(block1_up)
        print "conv ", conv._keras_shape

        output = Flatten()(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class Unet25D(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 12
        self.patchZ_out = 4
        self.patchSize = 508
        self.patchSize_out = 324
        self.filename = "unet_25d_dice"
        self.loss = dice_coef_loss

    def build(self):

        def crop_layer(x, cs, csz):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            csz = int(csz)
            if csz == 0:
                return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
            else:
                return x[:, :, csz:-csz, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, k1=1, k2=1, doPooling=True, downsampleZ=False):
            act1 = Conv3D(filters=filters, kernel_size=(k1,3,3), strides=(1,1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(input)

            act2 = Conv3D(filters=filters, kernel_size=(k2,3,3), strides=(1,1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)

            if doPooling:
                if downsampleZ:
                    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(act2)
                else:
                    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding="valid")(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out, k1=1, k2=1, upsampleZ=False):
                print "input ", input._keras_shape
                if upsampleZ:
                    up_sampled = UpSampling3D(size=(2,2,2))(input)
                else:
                    up_sampled = UpSampling3D(size=(1,2,2))(input)
                print "upsampled ", up_sampled._keras_shape

                conv_up = Conv3D(filters=filters, kernel_size=(2,2,2), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(up_sampled)
                print "up-convolution ", conv_up._keras_shape
                print "to be merged with ", down_block_out._keras_shape

                cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
                csZ = (down_block_out._keras_shape[2] - conv_up._keras_shape[2]) / 2.

                down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                                arguments={"cs": cropSize, "csz":csZ})(down_block_out)

                print "cropped layer size: ", down_block_out_cropped._keras_shape
                merged = concatenate([conv_up, down_block_out_cropped], axis=1)

                print "merged ", merged._keras_shape
                act1 = Conv3D(filters=filters, kernel_size=(k1,3,3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="valid",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(merged)

                print "conv1 ", act1._keras_shape
                act2 = Conv3D(filters=filters, kernel_size=(k2,3,3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="valid",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)
                print "conv2 ", act2._keras_shape

                return act2

        print
        print "==== building network ===="
        print
        input = Input(shape=(1, self.patchZ, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape

        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels, k1=2, k2=2, downsampleZ=False)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2, k1=1, k2=1, downsampleZ=False)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4, k1=2, k2=2, downsampleZ=True)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8, k1=1, k2=1, downsampleZ=True)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False, k1=1, k2=1)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act, upsampleZ=True, k1=1, k2=1)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act, upsampleZ=True, k1=2, k2=2)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act, upsampleZ=False, k1=1, k2=1)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act, upsampleZ=False, k1=2, k2=2)
        print
        print "== 1x1 convolution =="

        conv = Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1),
                        kernel_initializer=self.initialization, padding="valid", activation='sigmoid')(block1_up)

        output = Flatten()(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class Unet3D(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 16
        self.patchZ_out = 8
        self.patchSize = 350
        self.patchSize_out = 306
        self.numKernels = 16
        self.filename = "unet_3d"
        self.loss = weighted_mse

    def build(self):

        def retanh(x):
            return K.maximum(0, K.tanh(x))

        def unet_block_down(input, filters, doPooling=True, doDropout=False, doBatchNorm=False, downsampleZ=False, thickness1=1, thickness2=1):
            # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
            # All are valid area, not same
            act1 = Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                activation='relu', kernel_regularizer=l2(self.weight_decay))(input)
            #act1 = ReLU()(act1)

            if doBatchNorm:
                act1 = BatchNormalization(axis=1)(act1)

            act2 = Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="valid",
                                activation='relu', kernel_regularizer=l2(self.weight_decay))(act1)
            #act2 = ReLU()(act2)

            if doBatchNorm:
                act2 = BatchNormalization(axis=1)(act2)

            if doDropout:
                act2 = Dropout(0.5)(act2)

            if doPooling:
                # now downsamplig with maxpool
                if downsampleZ:
                    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(act2)

                else:
                    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(act2)

            else:
                pool1 = act2

            return (act2, pool1)


        # need to define lambda layer to implement cropping
        def crop_layer(x, cs, csZ):
            cropSize = cs
            if csZ == 0:
                return x[:,:,:,cropSize:-cropSize, cropSize:-cropSize]
            else:   
                return x[:,:,csZ:-csZ,cropSize:-cropSize, cropSize:-cropSize]


        def unet_block_up(input, filters, down_block_out, doBatchNorm=False, upsampleZ=False, thickness1=1, thickness2=1):
            print "This is unet_block_up"
            print "input ", input._keras_shape
            # upsampling
            if upsampleZ:
                up_sampled = UpSampling3D(size=(2,2,2))(input)
            else:
                up_sampled = UpSampling3D(size=(1,2,2))(input)
            print "upsampled ", up_sampled._keras_shape
            # up-convolution
            conv_up = Conv3D(filters=filters, kernel_size=(1, 2, 2), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                activation='relu', kernel_regularizer=l2(self.weight_decay))(up_sampled)
            #conv_up = PReLU()(conv_up)

            print "up-convolution ", conv_up._keras_shape
            # concatenation with cropped high res output
            # this is too large and needs to be cropped
            print "to be merged with ", down_block_out._keras_shape

            cropSize = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
            csZ      = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
            if cropSize>0:
            # input is a tensor of size (batchsize, channels, thickness, width, height)
                down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:], arguments={"cs":cropSize,"csZ":csZ})(down_block_out)
            else: 
                down_block_out_cropped = down_block_out
                
            print "cropped layer size: ", down_block_out_cropped._keras_shape
            merged = concatenate([conv_up, down_block_out_cropped], axis=1)

            print "merged ", merged._keras_shape
            act1 = Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                activation='relu', kernel_regularizer=l2(self.weight_decay))(merged)
            #act1 = ReLU()(act1)

            if doBatchNorm:
                act1 = BatchNormalization(axis=1)(act1)

            print "conv1 ", act1._keras_shape
            act2 = Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="valid",
                                activation='relu', kernel_regularizer=l2(self.weight_decay))(act1)
            #act2 = ReLU()(act2)

            if doBatchNorm:
                act2 = BatchNormalization(axis=1)(act2)

            print "conv2 ", act2._keras_shape

            return act2


        # input data should be large patches as prediction is also over large patches
        print
        print "==== building network ===="
        print

        print "== BLOCK 1 =="
        input = Input(shape=(1, self.patchZ, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels, doBatchNorm=True, thickness1=3, thickness2=3, downsampleZ=False)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2, doBatchNorm=True, thickness1=3, thickness2=3, downsampleZ=False)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4, doBatchNorm=True, thickness1=3, thickness2=3, downsampleZ=False)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8, doBatchNorm=True, thickness1=3, thickness2=3, doPooling=False, downsampleZ=False)
        print "block4 ", block4_pool._keras_shape

        ##print "== BLOCK 5 =="
        ##print "#no pooling for the bottom layer"
        ##block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False, doBatchNorm=True, thickness1=3, thickness2=3)
        ##print "block5 ", block5_pool._keras_shape

        ##print
        ##print "=============="
        ##print

        ##print "== BLOCK 4 UP =="
        ##block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act, doBatchNorm=True, upsampleZ=False, thickness1=3, thickness2=3)
        ##print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_act,  filters=self.numKernels*4, down_block_out=block3_act, doBatchNorm=True, upsampleZ=False, thickness1=3, thickness2=3)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act, doBatchNorm=True, upsampleZ=True,thickness1=3, thickness2=3)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act, doBatchNorm=True, upsampleZ=False,thickness1=3, thickness2=3)
        print
        print "== 1x1 convolution =="
        output = Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1),
                                kernel_initializer=self.initialization,
                                activation=retanh, padding="valid")(block1_up)

        print "output ", output._keras_shape
        output_flat = Flatten()(output)
        print "output flat ", output_flat._keras_shape
        print

        model = Model(inputs=input, outputs=output_flat)
        return model
        '''
        def retanh(x):
            return K.maximum(0, K.tanh(x))

        def crop_layer(x, cs, csz):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            csz = int(csz)
            if csz == 0:
                return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
            else:
                return x[:, :, csz:-csz, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                            kernel_initializer=self.initialization, padding="same",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(input)

            act2 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)

            if doPooling:
                pool1 = MaxPooling3D(pool_size=(1,2,2))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out, upsampleZ=False):
                print "input ", input._keras_shape
                if upsampleZ:
                    up_sampled = UpSampling3D(size=(2,2,2))(input)
                else:
                    up_sampled = UpSampling3D(size=(1,2,2))(input)
                print "upsampled ", up_sampled._keras_shape

                conv_up = Conv3D(filters=filters, kernel_size=(2,2,2), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(up_sampled)
                print "up-convolution ", conv_up._keras_shape
                print "to be merged with ", down_block_out._keras_shape

                cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
                csZ = (down_block_out._keras_shape[2] - conv_up._keras_shape[2]) / 2.

                down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                                arguments={"cs": cropSize, "csz":csZ})(down_block_out)

                print "cropped layer size: ", down_block_out_cropped._keras_shape
                merged = concatenate([conv_up, down_block_out_cropped], axis=1)

                print "merged ", merged._keras_shape
                act1 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(merged)

                print "conv1 ", act1._keras_shape
                act2 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="valid",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)
                print "conv2 ", act2._keras_shape

                return act2

        print
        print "==== building network ===="
        print
        input = Input(shape=(1, self.patchZ, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape

        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act, upsampleZ=True)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act, upsampleZ=True)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="

        conv = Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh)(block1_up)

        output = Flatten()(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model
        '''

class Unet3D_same(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 4
        self.patchZ_out = 4
        self.patchSize = 320
        self.patchSize_out = 320
        self.numKernels = 32
        self.filename = "unet_3d_same"
        self.loss = weighted_mse

    def build(self):

        def retanh(x):
            return K.maximum(0, K.tanh(x))

        def crop_layer(x, cs, csz):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            csz = int(csz)
            if csz == 0:
                return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
            else:
                return x[:, :, csz:-csz, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                            kernel_initializer=self.initialization, padding="same",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(input)

            act2 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                            kernel_initializer=self.initialization, padding="same",
                          kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)

            if doPooling:
                pool1 = MaxPooling3D(pool_size=(1,2,2))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out, upsampleZ=False):
                print "input ", input._keras_shape
                if upsampleZ:
                    up_sampled = UpSampling3D(size=(2,2,2))(input)
                else:
                    up_sampled = UpSampling3D(size=(1,2,2))(input)
                print "upsampled ", up_sampled._keras_shape

                conv_up = Conv3D(filters=filters, kernel_size=(1,2,2), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(up_sampled)
                print "up-convolution ", conv_up._keras_shape
                print "to be merged with ", down_block_out._keras_shape

                merged = concatenate([conv_up, down_block_out], axis=1)

                print "merged ", merged._keras_shape
                act1 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(merged)

                print "conv1 ", act1._keras_shape
                act2 = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1,1,1),
                                kernel_initializer=self.initialization, padding="same",
                                kernel_regularizer=l2(self.weight_decay), activation='relu')(act1)
                print "conv2 ", act2._keras_shape

                return act2

        print
        print "==== building network ===="
        print
        input = Input(shape=(1, self.patchZ, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape

        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="

        conv = Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1),
                        kernel_initializer=self.initialization, padding="same", activation=retanh)(block1_up)

        output = Flatten()(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model       

class RUnet(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 8
        self.patchZ_out = 8
        self.patchSize = 508
        self.patchSize_out = 322
        self.filename = "runet"
        self.loss = weighted_mse

    def build(self):

        def retanh(x):
            return K.maximum(0, K.tanh(x))

        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(input)
            print "act1", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "act2", act1._keras_shape

            if doPooling:
                pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = TimeDistributed(UpSampling2D(size=(2,2)))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = TimeDistributed(Conv2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
        
            print "cropped layer size: ", down_block_out_cropped._keras_shape

            merged = concatenate([conv_up, down_block_out_cropped], axis=2)
            
            print "merged ", merged._keras_shape
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(merged)

            print "conv1 ", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(None, 1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="
        
        convlstm = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                        kernel_initializer=self.initialization,
                        kernel_regularizer=l2(self.weight_decay), activation='relu',
                        padding='valid', return_sequences=True)(block1_up)
        print "convlstm ", convlstm._keras_shape
        
        conv = TimeDistributed(Conv2D(filters=3, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh))(convlstm)
        print "last conv ", conv._keras_shape
        
        output = TimeDistributed(Flatten())(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class RUnet_left(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 8
        self.patchZ_out = 8
        self.patchSize = 508
        self.patchSize_out = 324
        self.filename = "runet_left"
        self.loss = weighted_mse

    def build(self):
    
        def retanh(x):
            return K.maximum(0, K.tanh(x))
            
        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True, doRecurrent=False):
            if doRecurrent:
                act1 = ConvLSTM2D(filters=filters, kernel_size=(3, 3),
                            kernel_initializer=self.initialization,
                            kernel_regularizer=l2(self.weight_decay), activation='relu',
                            padding='valid', return_sequences=True)(input)
                print "convlstm 1"
                act2 = ConvLSTM2D(filters=filters, kernel_size=(3, 3),
                            kernel_initializer=self.initialization,
                            kernel_regularizer=l2(self.weight_decay), activation='relu',
                            padding='valid', return_sequences=True)(act1)
                print "convlstm 2"
            else:
                act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(input)
                print "conv2d 1"
                act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                                kernel_initializer=self.initialization, padding="valid",
                              kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
                print "conv2d 2"
            if doPooling:
                pool1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = TimeDistributed(UpSampling2D(size=(2,2)))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = TimeDistributed(Conv2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
        
            print "cropped layer size: ", down_block_out_cropped._keras_shape

            merged = concatenate([conv_up, down_block_out_cropped], axis=2)
            
            print "merged ", merged._keras_shape
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(merged)

            print "conv1 ", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(None, 1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8, doRecurrent=False)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False, doRecurrent=True)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="

        conv = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh))(block1_up)
        print "last conv ", conv._keras_shape
        
        output = TimeDistributed(Flatten())(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class RUnet_graft(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 8
        self.patchZ_out = 8
        self.patchSize = 508
        self.patchSize_out = 324
        self.filename = "runet_graft"
        self.loss = weighted_mse

    def build(self):
        '''
        def retanh(x):
            return K.maximum(0, K.tanh(x))

        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(input)
            print "act1", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "act2", act1._keras_shape

            if doPooling:
                pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = TimeDistributed(UpSampling2D(size=(2,2)))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = TimeDistributed(Conv2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
        
            print "cropped layer size: ", down_block_out_cropped._keras_shape

            merged = concatenate([conv_up, down_block_out_cropped], axis=2)
            
            print "merged ", merged._keras_shape
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(merged)

            print "conv1 ", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(None, 1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="
        
        # convlstm = ConvLSTM2D(filters=32, kernel_size=(3, 3),
        #                 kernel_initializer=self.initialization,
        #                 kernel_regularizer=l2(self.weight_decay), activation='relu',
        #                 padding='valid', return_sequences=True)(block1_up)
        # print "convlstm ", convlstm._keras_shape
        
        conv = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh))(block1_up)
        print "last conv ", conv._keras_shape
        '''

        input_model = model_from_json(open(self.base_filename + '.json').read())
        input_model.load_weights(self.base_filename + '_weights.h5')
        # for layer in input_model.layers:
        #     layer.trainable=False
        convlstm = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                        kernel_initializer=self.initialization,
                        kernel_regularizer=l2(self.weight_decay), activation='relu',
                        padding='valid', return_sequences=True)(input_model.layers[38].output)
        print "convlstm ", convlstm._keras_shape
        conv = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh))(block1_up)
        print "last conv ", conv._keras_shape               
        output = TimeDistributed(Flatten())(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input_model.input, outputs=output)
        return model

class RUnet_full(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 8
        self.patchZ_out = 8
        self.patchSize = 320
        self.patchSize_out = 228
        self.filename = "runet_full"
        self.numKernels = 16
        self.loss = weighted_mse

    def build(self):

        def retanh(x):
            return K.maximum(0, K.tanh(x))

        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
    
        def unet_block_down(input, filters, doPooling=True):
            act1 = ConvLSTM2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(input)
            print "act1", act1._keras_shape
            act2 = ConvLSTM2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(act1)
            print "act2", act1._keras_shape

            if doPooling:
                pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = TimeDistributed(UpSampling2D(size=(2,2)))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = ConvLSTM2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
        
            print "cropped layer size: ", down_block_out_cropped._keras_shape

            merged = concatenate([conv_up, down_block_out_cropped], axis=2)
            
            print "merged ", merged._keras_shape
            act1 = ConvLSTM2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(merged)

            print "conv1 ", act1._keras_shape
            act2 = ConvLSTM2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(None, 1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        #print "== BLOCK 5 =="
        #block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        #print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        #print "== BLOCK 4 UP =="
        #block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        #print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_act,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="
        
        conv = ConvLSTM2D(filters=1, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh, return_sequences=True)(block1_up)
        print "last conv ", conv._keras_shape
        
        output = TimeDistributed(Flatten())(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class RUnet_middle(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 8
        self.patchZ_out = 8
        self.patchSize = 508
        self.patchSize_out = 324
        self.filename = "runet_middle"
        self.loss = weighted_mse

    def build(self):
    
        def retanh(x):
            return K.maximum(0, K.tanh(x))
            
        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(input)

            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)

            if doPooling:
                pool1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = TimeDistributed(UpSampling2D(size=(2,2)))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = TimeDistributed(Conv2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
            print "cropped layer size: ", down_block_out_cropped._keras_shape

            down_block_out_lstm = ConvLSTM2D(filters=conv_up._keras_shape[2], kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                          kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(down_block_out_cropped)
            print "lstm layer size: ", down_block_out_lstm._keras_shape

            merged = concatenate([conv_up, down_block_out_lstm], axis=2)
            
            print "merged ", merged._keras_shape
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(merged)

            print "conv1 ", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(None, 1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="

        conv = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh))(block1_up)
        print "last conv ", conv._keras_shape
        
        output = TimeDistributed(Flatten())(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class RUnet_both(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 12
        self.patchZ_out = 12
        self.patchSize = 508
        self.patchSize_out = 322
        self.filename = "runet_both"
        self.loss = weighted_mse

    def build(self):
    
        def retanh(x):
            return K.maximum(0, K.tanh(x))
            
        def crop_layer(x, cs):
            odd = int(int(cs) != round(cs))
            cs = int(cs)
            return x[:, :, :, cs:-cs-odd, cs:-cs-odd]
        
        def unet_block_down(input, filters, doPooling=True):
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(input)

            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                          kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)

            if doPooling:
                pool1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(act2)
            else:
                pool1 = act2

            return (act2, pool1)

            return (act2, pool1)

        def unet_block_up(input, filters, down_block_out):
            print "input ", input._keras_shape
            up_sampled = TimeDistributed(UpSampling2D(size=(2,2)))(input)
            print "upsampled ", up_sampled._keras_shape

            conv_up = TimeDistributed(Conv2D(filters=filters, kernel_size=(2,2), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(up_sampled)
            print "up-convolution ", conv_up._keras_shape
            print "to be merged with ", down_block_out._keras_shape

            cropSize = (down_block_out._keras_shape[3] - conv_up._keras_shape[3]) / 2.
            
            down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:],
                                            arguments={"cs": cropSize})(down_block_out)
            print "cropped layer size: ", down_block_out_cropped._keras_shape

            down_block_out_lstm = ConvLSTM2D(filters=conv_up._keras_shape[2], kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="same",
                          kernel_regularizer=l2(self.weight_decay), activation='relu', return_sequences=True)(down_block_out_cropped)
            print "lstm layer size: ", down_block_out_lstm._keras_shape

            merged = concatenate([conv_up, down_block_out_lstm], axis=2)
            
            print "merged ", merged._keras_shape
            act1 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(merged)

            print "conv1 ", act1._keras_shape
            act2 = TimeDistributed(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),
                            kernel_initializer=self.initialization, padding="valid",
                            kernel_regularizer=l2(self.weight_decay), activation='relu'))(act1)
            print "conv2 ", act2._keras_shape

            return act2
        
        print
        print "==== building network ===="
        print
        input = Input(shape=(None, 1, self.patchSize, self.patchSize))
        print "input  ", input._keras_shape
        
        print "== BLOCK 1 =="
        block1_act, block1_pool = unet_block_down(input=input, filters=self.numKernels)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, filters=self.numKernels*2)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, filters=self.numKernels*4)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, filters=self.numKernels*8)
        print "block4 ", block4_pool._keras_shape

        print "== BLOCK 5 =="
        block5_act, block5_pool = unet_block_down(input=block4_pool, filters=self.numKernels*16, doPooling=False)
        print "block5 ", block5_pool._keras_shape

        print
        print "=============="
        print

        print "== BLOCK 4 UP =="
        block4_up = unet_block_up(input=block5_act, filters=self.numKernels*8, down_block_out=block4_act)
        print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_up,  filters=self.numKernels*4, down_block_out=block3_act)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  filters=self.numKernels*2, down_block_out=block2_act)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  filters=self.numKernels*1, down_block_out=block1_act)
        print
        print "== 1x1 convolution =="

        convlstm = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                        kernel_initializer=self.initialization,
                        kernel_regularizer=l2(self.weight_decay), activation='relu',
                        padding='valid', return_sequences=True)(block1_up)
        print "convlstm ", convlstm._keras_shape
        
        conv = TimeDistributed(Conv2D(filters=1, kernel_size=(1,1), strides=(1,1),
                        kernel_initializer=self.initialization, padding="valid", activation=retanh))(convlstm)
        print "last conv ", conv._keras_shape
        
        output = TimeDistributed(Flatten())(conv)
        print "output ", output._keras_shape
        print
        model = Model(inputs=input, outputs=output)
        return model

class Binary_VGG16(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 1
        self.patchZ_out = 1
        self.patchSize = 224   
        self.patchSize_out = 224
        self.filename = "binary_vgg16"
        self.loss = "binary_crossentropy"

    def build(self):
    
        vgg = VGG16(include_top=False, weights=None, input_tensor=None,
                        input_shape=(3, self.patchSize, self.patchSize), pooling="max")
        output = Dense(1, activation='sigmoid')(vgg.output)
        model = Model(inputs=vgg.input, outputs=output)
        return model

class Binary_Conv8(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 1
        self.patchZ_out = 1
        self.patchSize = 324  
        self.patchSize_out = 324
        self.filename = "binary_conv8"
        self.loss = "binary_crossentropy"

    def build(self):
    
        input = Input(shape=(1, self.patchSize, self.patchSize))
        conv1 = Conv2D(filters=self.numKernels, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(input)
        conv2 = Conv2D(filters=self.numKernels, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(filters=self.numKernels*2, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(pool1)
        conv4 = Conv2D(filters=self.numKernels*2, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(filters=self.numKernels*4, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(pool2)
        conv6 = Conv2D(filters=self.numKernels*4, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(conv5)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
        conv7 = Conv2D(filters=self.numKernels*8, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(pool3)
        conv8 = Conv2D(filters=self.numKernels*8, kernel_size=(3,3), strides=(1,1), padding="valid", activation='relu')(conv7)
        flatt = Flatten()(conv8)
        output = Dense(1, activation="sigmoid")(flatt)
        model = Model(inputs=input, outputs=output)
        return model

class Binary_Unet2D(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)
        self.patchZ = 1
        self.patchZ_out = 1
        self.patchSize = 508
        self.patchSize_out = 324
        self.filename = "binary_unet_2d"
        self.loss = "binary_crossentropy"

    def setBaseFilename(self, base_filename=""):
        self.base_filename = base_filename

    def build(self):
        input_model = model_from_json(open(self.base_filename + '.json').read())
        input_model.load_weights(self.base_filename + '_weights.h5')
        for layer in input_model.layers:
            layer.trainable=False
        flat = Flatten()(input_model.layers[14].output)
        dense = Dense(32, activation='relu')(flat)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=input_model.input, outputs=output)
        return model

def getModelClass(modelType):
    if modelType == "runet":
        modelClass = RUnet()
    elif modelType == "runet_graft":
        modelClass = RUnet_graft()
    elif modelType == "runet_full":
        modelClass = RUnet_full()
    elif modelType == "runet_left":
        modelClass = RUnet_left()
    elif modelType == "runet_middle":
        modelClass = RUnet_middle()
    elif modelType == "runet_both":
        modelClass = RUnet_both()
    elif modelType == "unet_2d":
        modelClass = Unet2D()
    elif modelType == "unet_25d":
        modelClass = Unet25D()
    elif modelType == "unet_3d":
        modelClass = Unet3D()
    elif modelType == "unet_3d_same":
        modelClass = Unet3D_same()
    elif modelType == "binary_vgg16":
        modelClass = Binary_VGG16()
    elif modelType == "binary_unet_2d":
        modelClass = Binary_Unet2D()
    elif modelType == "binary_conv8":
        modelClass = Binary_Conv8()
    return modelClass

def getModelClassFromFilename(filename):
    if filename == "runet":
        modelClass = RUnet()
    elif filename == "runet_graft":
        modelClass = RUnet_graft()
    elif filename == "runet_full":
        modelClass = RUnet_full()
    elif filename == "runet_left":
        modelClass = RUnet_left()
    elif filename == "runet_middle":
        modelClass = RUnet_middle()
    elif filename == "runet_both":
        modelClass = RUnet_both()
    elif filename == "unet_2d":
        modelClass = Unet2D()
    elif filename == "unet_25d":
        modelClass = Unet25D()
    elif filename == "unet_3d":
        modelClass = Unet3D()
    elif filename == "unet_3d_same":
        modelClass = Unet3D_same()
    elif filename == "binary_vgg16":
        modelClass = Binary_VGG16()
    elif filename == "binary_unet_2d":
        modelClass = Binary_Unet2D()
    elif filename == "binary_conv8":
        modelClass = Binary_Conv8()
    return modelClass

def weighted_mse(y_true, y_pred):
    epsilon=0.00001
    pos_frac = 0.75
    neg_frac = 0.25
    y_pred = K.clip(y_pred,epsilon, 1-epsilon)
    # per batch positive fraction, negative fraction (0.5 = ignore)
    pos_mask = K.cast(y_true > pos_frac, 'float32')
    neg_mask = K.cast(y_true < neg_frac, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[1:]), 'float32')
    pos_fracs = K.clip((K.sum(pos_mask)/num_pixels),0.01, 0.99)
    neg_fracs = K.clip((K.sum(neg_mask) /num_pixels),0.01, 0.99)

    #pos_fracs = maybe_print(pos_fracs, "positive fraction",do_print=True)
    weight_factor = 1.0
    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = 1.0 / (2 * pos_fracs)
    neg_weight = weight_factor * 1.0 / (2 * neg_fracs)

    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error)

    return K.mean(batch_weighted_mse)

def unet_crossentropy_loss_sampled(y_true, y_pred):
    # weighted version of pixel-wise crossrntropy loss function
    alpha = 0.6
    epsilon = 1.0e-5
    y_pred_clipped = T.flatten(T.clip(y_pred, epsilon, 1.0-epsilon))
    y_true = T.flatten(y_true)
    # this seems to work
    # it is super ugly though and I am sure there is a better way to do it
    # but I am struggling with theano to cooperate
    # filter the right indices
    indPos = T.nonzero(y_true)[0] # no idea why this is a tuple
    indNeg = T.nonzero(1-y_true)[0]
    # shuffle
    n = indPos.shape[0]
    indPos = indPos[srng.permutation(n=n)]
    n = indNeg.shape[0]
    indNeg = indNeg[srng.permutation(n=n)]
   
    # take equal number of samples depending on which class has less
    n_samples = T.cast(T.min([T.sum(y_true), T.sum(1-y_true)]), dtype='int64')
    # indPos = indPos[:n_samples]
    # indNeg = indNeg[:n_samples]

    total = np.float64(self.patchSize_out*self.patchSize_out*self.patchZ_out)
    loss_vector = ifelse(T.gt(n_samples, 0),
             # if this patch has positive samples, then calulate the first formula
             (- alpha*T.sum(T.log(y_pred_clipped[indPos])) - (1-alpha)*T.sum(T.log(1-y_pred_clipped[indNeg])))/total, 
             - (1-alpha)*T.sum(T.log(1-y_pred_clipped[indNeg]))/total )

    average_loss = T.mean(loss_vector)/(1-alpha)
    return average_loss

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)