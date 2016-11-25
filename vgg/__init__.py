#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
sys.path.append('/opt/render')

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import os
import h5py
from matplotlib import pyplot as plt

import theano
theano.config.openmp = True

from colorama import *
init(autoreset=True)

from NetVidSeg import NetVidSegClass

class featureVGGClass(NetVidSegClass):

    def VGG_16(weights_path=None):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(1000, activation='softmax'))


        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')
        return model


    def load_model(path='/opt/render/neural/weights/vgg16_weights.h5'):

        model = VGG_16(path)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        return model


    def load_images(segments, allParam):

        ims = []
        files = []
        for index,segParam in enumerate(segments):

            if segParam['duration']>allParam['audio_bpm']*1.1:

                imagePath=allParam['local_output']+'/'+segParam['path']+'.jpg'
                print (Fore.GREEN + str(index)+': '+str(imagePath))


                try:
                    im = cv2.resize(cv2.imread(imagePath), (224, 224)).astype(np.float32)
                    im[:,:,0] -= 103.939
                    im[:,:,1] -= 116.779
                    im[:,:,2] -= 123.68
                    im = im.transpose((2,0,1))
                    im = np.expand_dims(im, axis=0)
                    ims.append(im)
                except:
                    pass

            else:
              print(Fore.YELLOW + 'skip ' + segParam['duration'])

        images = np.vstack(ims)
        print images.shape

        return images


    def predict(segments, allParam):

        images = load_images(segments, allParam)

        model = load_model()

        out = model.predict(images)
        print out
        print out.shape

        return out



    def distances(segments, allParam, images_predicted):

        from sklearn.metrics.pairwise import pairwise_distances

        print (Fore.YELLOW + '---------- distances -------------------' )

        dist = pairwise_distances(images_predicted[0],images_predicted, metric='cosine', n_jobs=1)


        top = np.argsort(dist[0])

        print top

###################################################################################################
if __name__ == '__main__':

    obj = feature_vgg()
    images_predicted = obj.predict()

    # distances(image_id)
