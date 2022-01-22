"""
Vison Based Control of Drones Project 
1/21/2022

Dr. Xiang
Patrick
Wesley 

model.py
"""
import tensorflow as tf
import numpy as np
import cv2
## object_detection API imports 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

class Model:
    """
        Class to load a tensorflow model.

        Parameters:
        
        a_model_path (string):
            The path to the "saved_ model folder
            obtained from exporting a tensorflow model
        
        a_label_path (string):
            The path to the label_map.pbtxt file
            This file contains the labels for each
            class the model is trained to detect. 
    """

    def __init__(self, a_model_path, a_label_path):
        """ Set everything up to run """
        # Suppress TensorFlow logging (2)
        tf.get_logger().setLevel('ERROR')           

        # enable dynamic memory 
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load the Model 
        self.detect_fn =  tf.saved_model.load(a_model_path)
        
        self.category_index = label_map_util.create_category_index_from_labelmap(a_label_path, use_display_name=True)

        self.imwidth = 640
        self.imheight = 480
    

    def detect(self, img, max_boxes, min_score):
        """
            Method to utilize the model and make predictions 
        """
        
        # Prepare the image 
        img_np = np.array(img)
        input_tensor = tf.convert_to_tensor(img_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Get output from model 
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))

        # unrap the model to make it easier to work with 
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_np_with_detections = img_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw= max_boxes,
            min_score_thresh= min_score,
            agnostic_mode=False)

        return image_np_with_detections, detections

    def draw_centers(self, img, detections, num_boxes):

        centers = []
        count = 0

        for box in detections['detection_boxes'][0:num_boxes]: # Only look at the top few we want to display 
            
            # Normalized coordinates from the model
            (ymin, xmin, ymax, xmax) = tuple(box.tolist())
            # Actual coordinates on the image 
            (left, right, top, bottom) = (xmin * self.imwidth, xmax * self.imwidth, ymin * self.imheight, ymax * self.imheight)

            xavg = (left + right) // 2 
            yavg = (bottom + top) // 2
            
            # The # that the model thinks this detectio is.
            # Corresponds to the label_map
            class_id = detections['detection_classes'][count]

            # A tuple containing the cordinates and the detection is added to a list
            centers.append((xavg, yavg, class_id))

            # Fancy stuff based on id
            if class_id == 1:
                img = cv2.circle(img, (int(xavg), int(yavg)), 4, (0, 0, 255), -1)
            else:
                img = cv2.circle(img, (int(xavg), int(yavg)), 4, (0, 0, 255), -1)
                # img = cv2.rectangle(img, (int(left), int(bottom)), (int(right), int(top)), (0,255,0), -1)
            
            count += 1

        return img, centers