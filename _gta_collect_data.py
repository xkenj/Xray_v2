# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
from grabscreen import grab_screen
import cv2
import keys as k
from getkeys import key_check
import time

# !!!!!!!!! A CHANGER EN FONCTION DU NEXT FICHIER NPY !!!!!!
starting_value = 1278
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

key_map = {
    'W': [1, 0, 0, 0, 0],
    'A': [0, 1, 0, 0, 0],
    'D': [0, 0, 1, 0, 0],
    'AW': [0, 0, 0, 1, 0],
    'DW': [0, 0, 0, 0, 1],
    'NK': [0, 0, 0, 0, 0]
}

training_data = []

keys = k.Keys({})

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")




# ******************************************************************************************************
                                #<<<< GTA >>>>>
# ******************************************************************************************************

def bluring(img):
    """Returns the blured image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to gray to reduce computation calculus
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # Average Average Filter to smooth the image. Often 5x5
    return blur


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    0  1  2  3  4   5   6   7    8
    [Z, Q, D, QZ, DZ,NOKEY] boolean values.
    '''
    if ''.join(keys) in key_map:
        return key_map[''.join(keys)]
    return key_map['NK']
    

for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)      
print('STARTING!!!')

paused = False
last_time = time.time()
while True:   
    if not paused:                                     #800,640
        # screen = cv2.resize(grab_screen(region=(0, 40, 800, 640)), (800, 450))  # To modify
        #1152 x 864
        screen = grab_screen(region=(0, 150, 800, 590))

        image_np = cv2.resize(screen, (180, 133))
                
        # ********************* Collect Data ********************************

        keys_check = key_check()
        output = keys_to_output(keys_check)
        training_data.append([image_np, output])

        file_name = 'frames/training_data-{0}.npy'.format(starting_value)
        if len(training_data) % 100 == 0:
            print(len(training_data))
            if len(training_data) % 1000 == 0:
                np.save(file_name, training_data)
                print('SAVED')
                training_data = []
                starting_value += 1
                file_name = 'frames/training_data-{0}.npy'.format(starting_value)
        
    pause_check = key_check()
    if 'T' in pause_check:
        if paused:
            paused = False
            print('Go!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)

    # print('Loop took {} seconds'.format(time.time() - last_time))  # Check the FPS (print impact the fps)
    # last_time = time.time()
    # **************************************************************

    # cv2.imshow('window', image_np)
    # cv2.imshow('window2', screen_saved)
    if cv2.waitKey(25) & 0xff == ord('p'):
        cv2.destroyAllWindows()
        break
    