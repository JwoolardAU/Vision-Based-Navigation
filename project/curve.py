from ast import While
import cv2
import concurrent.futures
from model import Model
from djitellopy import Tello
from djitellopy import TelloSwarm

def Tello_Script():
    swarm.takeoff()
    swarm.parallel(lambda i, tello: tello.curve_xyz_speed(25, -25, 0, 25, -75, 0, 20))
    swarm.land()

PATH_TO_SAVED_MODEL = r'.\resources\model\saved_model'
PATH_TO_LABELS = r'.\resources\annotations\label_map.pbtxt'

# How many boxes do we expect? 
max_boxes = 6
# How confident does the model need to be to display any bouding box? 
min_score_thresh = .70
cam = cv2.VideoCapture(1)
Tf_Model = Model(PATH_TO_SAVED_MODEL, PATH_TO_LABELS, cam, max_boxes, min_score_thresh)

drone = Tello('192.168.1.200')
drone2 = Tello('192.168.1.100')

swarm = TelloSwarm([drone,drone2])
swarm.connect()

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(Tf_Model.update_bounding_boxes_thread)
    executor.submit(Tello_Script)

    while True:
        cv2.imshow("Live_Stream", Tf_Model.bounding_box_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()












