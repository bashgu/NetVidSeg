#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os,sys
import numpy as np
import json

sys.path.append('/opt/render')
from tools import mysql
from tools import bash

from colorama import *
init(autoreset=True)

from tools.decorators import *

#################################################################################

class NetVidSegClass(object):

    """
    NetVidSeg - Net for Video Segmentations
    Общий класс для вытаскивания признаков c помощью нейронок
    """

    def __init__(self):
        import NetVidSeg
        # self.allParam = allParam
        return

    # @decor_function_call
    # def extract(self):
    #     return
    #
    # @decor_function_call
    # def load_model(self):
    #     return
    #
    # @decor_function_call
    # def load_images(self):
    #     return


    @decor_function_call
    def save_to_db(self, segments, allParam, test=False):
      from collections import Counter

      #инициализация сетки

    #   images_predicted = neural.ssd()
    #   images_predicted = vgg.keras_vgg16.predict(segments)

    #   vgg.distances(segments, allParam, images_predicted)


      # classSum=[]
      #
      # for index,segParam in enumerate(segments):
      #   print (segParam)
      #
      #   if segParam['duration']>allParam['audio_bpm']*1.1:
      #       #начинаем распознование если не короткий
      #
      #
      #
      #       id_segm = segParam['id']
      #
      #       recParam={
      #       'id_file':str(segParam['id_file']),
      #       'id_segm':str(id_segm),
      #       'status':'0',
      #       'img':str(segParam['path']+'.jpg'),
      #       'type':'googlenet',
      #       'class':str(predicted_class),
      #       #'metadata':str(labels[predicted_class]).replace("'","\\'")
      #       # 'metadata':'{"name":"'+str(labels[predicted_class]).replace("'","\\'")+'"}'
      #             }
      #
      #       classSum.append(recParam['class'])
      #
      #
      #       #print (Fore.YELLOW + '> ' + str(recParam['type']) + ': ' + Fore.CYAN + str(predicted_class)+' --> ' +str(labels[predicted_class]))
      #
      #       if not test:
      #       #  mysql.delRecogn(id_segm)
      #       #  mysql.insRecogn(recParam)
      #           pass
      #
      #   else:
      #     print(Fore.YELLOW + 'skip ' + segParam['duration'])



      return


# --------------------------------------------------------------------


if __name__ == '__main__':


    print '---------------------------------'

    obj = NetVidSegClass()

    obj.detect_object_on_image()
