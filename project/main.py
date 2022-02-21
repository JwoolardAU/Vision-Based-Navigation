"""
Vison Based Control of Drones Project 
1/21/2022

Dr. Xiang
Patrick
Wesley 

main.py
"""
import sys
import time
from cv2 import CAP_PROP_EXPOSURE
import tensorflow as tf
import numpy as np
import cv2
## object_detection API imports 
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
## Custom Classes
from model import Model

def fix_color(img):
    updated_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return updated_img


cam = cv2.VideoCapture(0)

PATH_TO_SAVED_MODEL = r'.\resources\model\saved_model'
PATH_TO_LABELS = r'.\resources\annotations\label_map.pbtxt'

Detector = Model(PATH_TO_SAVED_MODEL, PATH_TO_LABELS)

# Drone 1 has password of 12345678 



print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))





# How many boxes do we want displaying? 
num_boxes = 10

# How confident does the model need to be to display any bouding box? 
min_score_thresh = .70

# cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
# cam.set(cv2.CAP_PROP_EXPOSURE, -2)

for x in range(100):
    _, img = cam.read()

cv2.imshow('im1', img)
cv2.waitKey(0)

while True:
    _, img = cam.read()

    img = fix_color(img)

    flag_centers = Detector.get_one_class_centers(img, 4, 2)

    print(flag_centers)
    if flag_centers:
        flag_center = flag_centers[0]
        flag_center = (int(flag_center[0]), int(flag_center[1]))
    else:
        continue


    # get old positions
    pre_moving = Detector.IDS.get_current_positions()

    # Move the drones
    print('You have 5 seconds to move the drone')
    time.sleep(5)

    _, img2 = cam.read()
    img2 = fix_color(img2)

    img2 = np.array(img2)
    detections = Detector.detect(img2)
    Detector.update_ids(detections, num_boxes)

    # get new positions 
    post_moving = Detector.IDS.get_current_positions()

    print(pre_moving)
    print(post_moving)
    # compute the vectors



    flag1_vec = Detector.compute_2d_vector(flag_centers[0], post_moving[0]) # center - post 
    flag2_vec = Detector.compute_2d_vector(flag_centers[0], post_moving[1])

    drone1dir = Detector.compute_2d_vector(post_moving[0], pre_moving[0]) # Post - Pre
    drone2dir = Detector.compute_2d_vector(post_moving[1], pre_moving[1]) 

    drone1_angle = Detector.dotProduct_2d(drone1dir, flag1_vec) 
    drone2_angle = Detector.dotProduct_2d(drone2dir, flag1_vec)

    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    img2 = cv2.circle(img2, flag_center, 3, (0,255,0), thickness=-1)
    img2 = cv2.circle(img2, pre_moving[0], 3, (0,0,255), thickness=-1)
    img2 = cv2.circle(img2, post_moving[0], 3, (0,0,255), thickness=-1)

    # Direction
    output = cv2.line(img2, post_moving[0],(post_moving[0][0] + drone1dir[0], post_moving[0][1] + drone1dir[1]), (255,0,255), thickness = 3)
    output = cv2.line(output, post_moving[1],(post_moving[1][0] + drone2dir[0], post_moving[1][1] + drone2dir[1]), (255,0,255),thickness = 3)

    # Vector to flag
    output = cv2.line(output,  post_moving[0],(post_moving[0][0] + flag1_vec[0], post_moving[0][1] + flag1_vec[1]), (255,0,0),thickness = 3)
    output = cv2.line(output,  post_moving[1],(post_moving[1][0] + flag2_vec[0], post_moving[1][1] + flag2_vec[1]), (255,0,0),thickness = 3)

    cv2.imshow("output", output) 
    cv2.waitKey(0)



# compute the angles :D 


while True:
    ret, img = cam.read()

    if ret:
        # Convert BGR to RGB.
        color_correct = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Send the picture to the model.
        detected_image, detections = Detector.draw_bounding_boxes(color_correct, num_boxes, min_score_thresh)

        # Convert RGB back to BGR for cv2's imshow function.
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)

        # update the ID's
        Detector.update_ids(detections, num_boxes)

        for id in Detector.IDS.IDS:
            detected_image = id.draw_history(20, detected_image)
        
        updated_image = Detector.IDS.draw_ids(detected_image)

        # Get the num_boxes number of center coordinates and display them on the image. 
        #centers_img, drone_centers, flag_centers = Detector.draw_centers(detected_image, detections, num_boxes)

        cv2.imshow('Img', updated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

"""
What's next?? 

Determine Fly to Points 

Associate Drone Coordinates wtih ID's 

Determine drone direction.
"""

'''
Ideas: 
- Save all paper points once at the beginning of the program
- Use paper points as fly-to-points
- Update paper positions after drone has completed a flight command (i.e. 'checkpoint')
'''
# python exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\exported-models\all_data\pipeline.config --trained_checkpoint_dir .\exported-models\all_data --output_directory .\exported-models\model_final
