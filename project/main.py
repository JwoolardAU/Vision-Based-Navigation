import cv2
import tensorflow as tf
import numpy as np
import time


from model import Model

PATH_TO_SAVED_MODEL = r'.\resources\model\saved_model'
PATH_TO_LABELS = r'.\resources\annotations\label_map.pbtxt'

# How many boxes do we expect? 
max_boxes = 6
# How confident does the model need to be to display any bouding box? 
min_score_thresh = .70

cam = cv2.VideoCapture(1)

Tf_Model = Model(PATH_TO_SAVED_MODEL, PATH_TO_LABELS, cam, max_boxes, min_score_thresh)

## First detections and centers have been computed in Model Set up

### Assign the same flag to each ID for now ## 
for id in Tf_Model.id_manager.IDS:
    id.flag_point = Tf_Model.flag_centers[0]

# cv2.imshow("test1", cv2.cvtColor(Tf_Model.img, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #######################################################################
# Get initial Direction of the drone 

while True:

    ## Set the starting positions for use later
    Tf_Model.id_manager.set_starting_positions()

    ## Update information for model
    Tf_Model.updateDetections()
    Tf_Model.update_centers(max_boxes)

    ## Set the ending position
    Tf_Model.id_manager.set_ending_positions()


    ## Compute the vectors and turn angles 
    Tf_Model.computeDirectionVectors()
    Tf_Model.computeFlagVector()
    Tf_Model.computeTurnAngles()

    img = Tf_Model.id_manager.draw_vectors(Tf_Model.img)
    img = Tf_Model.id_manager.draw_id_nums(img)
    img = Tf_Model.draw_bounding_boxes(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print(f"Id 1 angle: {Tf_Model.id_manager.IDS[0].turn_angle}")
    print(f"Id 2 angle: {Tf_Model.id_manager.IDS[1].turn_angle}")

    cv2.imshow("Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#######################################################################

# while True:

#     Tf_Model.updateDetections()
#     Tf_Model.update_centers(max_boxes)

#     img = Tf_Model.draw_bounding_boxes()
#     img = Tf_Model.id_manager.draw_IDS_history(img, 10)
#     img = Tf_Model.id_manager.draw_id_nums(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     cv2.imshow("Img", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break





