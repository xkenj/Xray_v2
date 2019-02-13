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
from glob import glob

# !!!!!!!!! A CHANGER EN FONCTION DU NEXT FICHIER NPY !!!!!!
starting_value = 1241
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

FILES = "frames/Train3/*"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:

    saved = False

    while not saved:   
                                    #800,640
        # screen = cv2.resize(grab_screen(region=(0, 40, 800, 640)), (800, 450))  # To modify
        #1152 x 864

        for gta_file in glob(FILES):
            
            try:
                data = np.load(gta_file)
                screens = data[:,0]
                labels = data[:,1]

                training_data = []
                for index, screen in enumerate(screens):
                    screen = cv2.resize(screen, (107, 60))   #80
                    image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        skip_scores=True)
                    
                    # vehicule_dict = {}

                    for i, b in enumerate(boxes[0]):
                                            #car                  #bus                  #Truck
                        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8: # compute the number of pixels between x1 and x2 
                            if scores[0][i] > 0.6:  # As long as the score is above 50% (confident it is a car)
                                # Where is the middle point of the car ? 
                                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2 # bottom-left + bottom_right
                                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2 # top-left + topp_right                                       
                                #  Get more Granularity as we get closer
                                warning_distance = round( (1 - (boxes[0][i][3] - boxes[0][i][1]))**4, 1) # Get Smaller as Closer (Height of the boxes)
                                # stolen_distance = round( (1 - (boxes[0][i][3] - boxes[0][i][1]))**4, 1) # For car stolen, Don't flag too much
                                # vehicule_dict[stolen_distance] = [mid_x, mid_y, scores[0][i]]
                                cv2.putText(image_np, '{} m'.format(
                                    warning_distance*10), (int(mid_x * 800), int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                                # WARNING:  
                                if warning_distance <= 0.4:
                                    if mid_x > 0.3 and mid_x < 0.7:
                                        cv2.putText(image_np, '{}'.format(
                                    'WARNING !!!'), (int(mid_x * 800) - 50, int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            

                    # ********************* Collect Data ********************************

                    screen_saved = np.copy(image_np)
                    
                    screen_saved = bluring(screen_saved)
                    # # run a color convert:
                    # # screen_saved = cv2.cvtColor(screen_saved, cv2.COLOR_BGR2GRAY)  # !!! COLOR !!!
                    # keys_check = key_check()
                    # output = keys_to_output(keys_check)
                    training_data.append([screen_saved, labels[index]])
                # FILE2 = "frames/training_data-2050_tf.npy"
                np.save(gta_file, training_data)
                print('SAVED')

            except:
                pass
        saved = True
        print('END SAVING')

        
            



    