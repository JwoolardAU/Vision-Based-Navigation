from grpc import compute_engine_channel_credentials
import tensorflow as tf
import numpy as np
import cv2
import math 

# object_detection API Imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from new_idmanager import IDManager

class Model:
    """
        Class that will load a tensorflow Model
        and manage the camera connection. 

        The model requires the model path, the label map path,
        and a cv2 camera object. 

        Pass True for low_memory if your system has less memory. 
    """

    def __init__(self, a_model_path, a_label_path, camera_object, max_boxes, min_score_thresh, low_memory = False):
        """
            Set up everything to run
        """
        self.max_boxes = max_boxes
        self.thresh = min_score_thresh

        # Suppress TensorFlow logging 
        tf.get_logger().setLevel('ERROR')

        if low_memory: 
            # enable dynamic memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load the Model
        self.detector_function = tf.saved_model.load(a_model_path) 
        self.category_index = label_map_util.create_category_index_from_labelmap(a_label_path, use_display_name=True)
        self.drone_category = 1
        self.flag_category = 2

        # Set up the camer for use in the class
        self.cam = camera_object
        self.imwidth = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.imheight = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create ID manager to track Independent objects the model detects
        self.id_manager = IDManager()

        # Create an Initial ID for each drone used during flight
        self.id_manager.createID((20,20),(0,0,255))

        # Create attritubes for commonly accessed items so they don't
        # have to be juggled around Method calls
        _, img = self.cam.read()
        self.setImg(img)
        self.detections = self.getNewDetections()

        self.flag_centers = []
        self.drone_centers = []
        self.update_centers(self.max_boxes)

    def setImg(self, img):
        """
            Function to prepare the img attribute
        """
        color_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.array(color_fixed)
        self.img = img_np

        return None
    
    def updateImg(self):
        """
            Function to update the image attritube
            using the camera attribute
        """
        _, img = self.cam.read()
        self.setImg(img)

        return None
    
    def updateDetections(self):
        """
            Function to update the detections attritube
            with a new picture 
        """
        self.updateImg()
        self.detections = self.getNewDetections()

        return None

    def getNewDetections(self):
        """
            Method to utilize the model and make predictions
        """
        # Prepare the img
        input_tensor = tf.convert_to_tensor(self.img)
        input_tensor = input_tensor[tf.newaxis, ...]

        # get output from the model
        detections = self.detector_function(input_tensor)
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        return detections
    
    def draw_bounding_boxes(self, img = None):
        """
            Implementation of the object_detection API box drawer.
            Function will draw the bounding boxes of detctions on an image.

            Optional: Pass in your own img. 
            Must be a numpy array in RGB. 

            If no image is passed, a copy of self.img will be used.
        """
        if img is None:
            drawn_img = self.img.copy()
        else:
            drawn_img = img

        self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)
        viz_utils.visualize_boxes_and_labels_on_image_array(
            drawn_img,
            self.detections['detection_boxes'],
            self.detections['detection_classes'],
            self.detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw= self.max_boxes,
            min_score_thresh= self.thresh,
            agnostic_mode=False)

        return drawn_img
    
    def draw_detected_centers(self, img = None):
        """
            Funciton to draw what centers were detected.

            Optional: Pass in your own img. 
            Must be a numpy array in RGB. 

            If no image is passed, a copy of self.img will be used.
        """
        if img is None:
            drawn_img = self.img.copy()
        else:
            drawn_img = img

        for d_center, f_center in self.drone_centers, self.flag_centers:
            drawn_img = cv2.circle(drawn_img, d_center, radius=4, color=(0,255,255), thickness=-1)
            drawn_img = cv2.circle(drawn_img, f_center, radius=4, color=(0,255,255), thickness=-1)

        return drawn_img
    
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
        return (int(xavg), int(yavg))
    
    def update_centers(self, max_num_detections):
        """
            Function to get the flags center from detections.
            Will only look at max_num_detections number of detections. 
        """
        drone_cents = []
        flag_cents = []

        index = 0
        # Only look at the top few we specifiy. 
        for box in self.detections['detection_boxes'][0:max_num_detections]:
            
            # Computer the average centers cords
            center = self.compute_center(box)

            # Determine the cateogry and append it to the appropriate list
            class_category = self.detections['detection_classes'][index]
            if class_category == self.drone_category:
                drone_cents.append(center) 
            elif class_category == self.flag_category:
                flag_cents.append(center)
        
        self.drone_centers = drone_cents
        self.flag_centers = drone_cents

        self.id_manager.updatePositions(drone_cents)

        return None 
    
    def compute_2d_vector(self, point2, point1):
        """
            Function to compute a 2d vector given two points
        """
        (x2, y2) = point2
        (x1, y1) = point1

        vector = (int(x2-x1), int(y2-y1))

        return vector
    
    def dot_product(self, vector1, vector2):
        """
            Function to compute the dot product of two vectors
        """
        (x1, y1) = vector1
        (x2, y2) = vector2 

        dot = x1*x2 + y1*y2 

        return dot

    def determinate(self, vector1, vector2):
        """
            Function to compute a determinate
        """
        (x1,y1) = vector1
        (x2,y2) = vector2

        det = x1*y2 - y1*x2 

        return det
    
    def computeDirectionVectors(self): 
        """
            Function to compute the direction vectors for each ID
        """

        for id in self.id_manager.IDS:
            id.dir_vector = self.compute_2d_vector(id.ending_position, id.starting_position)
        
        return None
    
    def computeFlagVector(self):
        """
            Function to compute the direction vector to the appropriate flag for each ID
        """

        for id in self.id_manager.IDS:
            id.flag_vector = self.compute_2d_vector(id.flag_point, id.ending_position)

        return None

    def computeTurnAngles(self):
        """
            Function to compute the angle a Drone needs to 
            turn to face a flag
        """

        for id in self.id_manager.IDS:
            dot = self.dot_product(id.dir_vector, id.flag_vector)
            det = self.determinate(id.dir_vector, id.flag_vector)
            id.turn_angle = math.atan2(det, dot)

        return None

        