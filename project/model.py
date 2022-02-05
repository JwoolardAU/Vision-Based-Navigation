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

## ID manager import and plotting modules
from idmanager import IDManager
import matplotlib.pyplot as plt

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
	
	# Pat and Wes: Comment out for more VRAM, uncomment for machines w/ less resources
        # enable dynamic memory 
        #gpus = tf.config.experimental.list_physical_devices('GPU')
        #for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load the Model 
        self.detect_fn =  tf.saved_model.load(a_model_path)
        
        self.category_index = label_map_util.create_category_index_from_labelmap(a_label_path, use_display_name=True)

        self.imwidth = 640
        self.imheight = 480


        # Create ID manager in model and update positions in draw_centers
        # Since localization info for Drones is obtained there
        self.IDS = IDManager();

        # Create an ID for each drone used during flight
        self.IDS.createID((0,0)) # Drone 1 hard-coded for now
        self.IDS.createID((100,100)) # Drone 2 Hard-coded for now
        #self.IDS.createID((100,100)) # Drone 2 hard-coded for now

        self.dronepath = []

    def detect(self, img_np):
        """
            Method to utilize the model and make predictions 
        """
        
        # Prepare the image 
        input_tensor = tf.convert_to_tensor(img_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Get output from model 
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        return detections

    def draw_bounding_boxes(self, img, max_boxes, min_score):
        """ 
            Implementation of the object_detection API boxes 
        """
        img_np = np.array(img)
        detections = self.detect(img_np)

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
    
    def compute_center(self, box):
        """
            Method to compute the center of a box.
            Box is the normalized coords from the model. 
        """
        # Normalized coordinates from the model
        (ymin, xmin, ymax, xmax) = tuple(box.tolist())
        # Actual coordinates on the image 
        (left, right, top, bottom) = (xmin * self.imwidth, xmax * self.imwidth, ymin * self.imheight, ymax * self.imheight)

        xavg = (left + right) // 2 
        yavg = (bottom + top) // 2
            
        # Append a tuple containing the cordinates to the list of centers
        return (xavg, yavg)


    def get_class_centers(self, detections, num_boxes):
        """
            Method to parse detections -- the output of the model -- 
            to get the list of drone points and flag points separately. 
        """
        drone_centers = []
        flag_centers = []
        count = 0
        
        for box in detections['detection_boxes'][0:num_boxes]: # Only look at the top few we want to display 
            
            # The # that the model thinks this detection is.
            # Corresponds to the label_map
            class_id = detections['detection_classes'][count]

            if class_id == 1: # Drone has class 1, based on label_map.pbtxt
               center = self.compute_center(box)
               drone_centers.append(center)
            elif class_id == 2:
                center = self.compute_center(box)
                flag_centers.append(center)
            count += 1

        return drone_centers, flag_centers

    
    def draw_centers(self, img, detections, num_boxes):

        drone_centers, flag_centers = self.get_class_centers(detections, num_boxes)
        
        for center in drone_centers:
            (xavg, yavg) = center
            img = cv2.circle(img, (int(xavg), int(yavg)), 4, (0, 0, 255), -1) # b g r

        for center in flag_centers:
            (xavg, yavg) = center
            img = cv2.circle(img, (int(xavg), int(yavg)), 4, (0, 255, 255), -1)
            # img = cv2.rectangle(img, (int(left), int(bottom)), (int(right), int(top)), (0,255,0), -1)
        

        for id in self.IDS.IDS:
            print(f"Drone {id.IDnum} was at {id.get_position()}")
            self.dronepath.append(id.get_position())

        self.IDS.updatePositions(drone_centers)


        for id in self.IDS.IDS:
            print(f"ID {id.IDnum} is 'now' at {id.get_position()}")

        img = self.IDS.draw_ids(img)

        return img, drone_centers, flag_centers
