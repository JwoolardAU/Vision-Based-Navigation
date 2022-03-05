import cv2
import sys
import numpy as np
import tensorflow as tf
from model import Model
import math



# Hard code some points for now, Figure out how to allocate the points later. 

def twoDim_dot_product_theta(vector1, vector2):
    """
        Function to compute a dot product
        a dot b = |a||b|cos(theta) ->
        theta = acos( a dot b / (|a||b|) )
    """
    (x1, y1) = vector1
    (x2, y2) = vector2 

    a_dot_b = x1*x2 + y1*y2 
    length_a = math.sqrt(x1**2 + y1**2)
    length_b = math.sqrt(x2**2 + y2**2)

    theta = math.acos(a_dot_b/(length_a*length_b)) * (180/math.pi)

    return theta 

vector1=(2,4)
vector2=(3,2)

theta = twoDim_dot_product_theta(vector1, vector2)
print(theta)



# cv2.imshow('img', new)
# cv2.waitKey(0)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     sys.exit()

