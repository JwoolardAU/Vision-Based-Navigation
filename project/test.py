import cv2
import sys
import numpy as np
import tensorflow as tf
from model import Model
import math
import concurrent.futures


PATH_TO_SAVED_MODEL = r'.\resources\model\saved_model'
PATH_TO_LABELS = r'.\resources\annotations\label_map.pbtxt'

# How many boxes do we expect? 
max_boxes = 6
# How confident does the model need to be to display any bouding box? 
min_score_thresh = .50

cam = cv2.VideoCapture(0)

Tf_Model = Model(PATH_TO_SAVED_MODEL, PATH_TO_LABELS, cam, max_boxes, min_score_thresh)

with concurrent.futures.ThreadPoolExecutor() as executor:

    executor.submit(Tf_Model.basic_update_thread)
    executor.submit(Tf_Model.basic_img_thread)

    while True:
        cv2.imshow('img', Tf_Model.bounding_box_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    Tf_Model.cam.release()
    executor.shutdown(wait=False)


