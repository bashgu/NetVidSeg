#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os,sys

sys.path.append('/opt/render')

if __name__ == '__main__':

    from tools import mysql
    from tools import bash

    from colorama import *
    init(autoreset=True)

    import shutil, time

    if len(sys.argv)==0: os._exit()

    id_segm = sys.argv[1]
    print (id_segm)

    (segments, segDur) = mysql.selectSegments(id_segm)
    segParam = segments[0]
    print(segParam)

    (dir, file) = os.path.split(segParam['path'])
    (fileBaseName, fileExtension)=os.path.splitext(file)

    command="rclone copy --include '*.jpg' ibm:"+str(segParam['container'])+"/"+dir+" /tmp/"+dir

    bash.runCMD(command)

    allParam={}
    allParam['local_output']='/tmp'
    allParam['audio_bpm']=1

    feature_extract(segments, allParam, True)

    # keras_vgg16.distances('oE6boCQrzU8', files, images_predicted)


    # time.sleep(10)

    # shutil.rmtree('/tmp/'+dir)
