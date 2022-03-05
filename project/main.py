import cv2
import tensorflow as tf
import numpy as np
import time
import concurrent.futures

from model import Model

PATH_TO_SAVED_MODEL = r'.\resources\model\saved_model'
PATH_TO_LABELS = r'.\resources\annotations\label_map.pbtxt'

# How many boxes do we expect? 
max_boxes = 6
# How confident does the model need to be to display any bouding box? 
min_score_thresh = .70

cam = cv2.VideoCapture(0)

Tf_Model = Model(PATH_TO_SAVED_MODEL, PATH_TO_LABELS, cam, max_boxes, min_score_thresh)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(Tf_Model.update_vectors_thread)

    while True:
        cv2.imshow("Vector", Tf_Model.bounding_box_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean Exit
    cv2.destroyAllWindows()
    Tf_Model.cam.release()
    executor.shutdown(wait=False)