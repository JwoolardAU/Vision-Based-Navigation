import cv2
import sys
import numpy as np

cam = cv2.VideoCapture(1)
ret, img = cam.read()

#blue, green, red = cv2.split(img)

mask = np.ones((480,640,3), dtype=np.uint8)

mask = cv2.circle(mask, (100,100), 10, (0,0,0), -1)

mask = cv2.circle(mask, (200,200), 10, (0,0,0), -1)

#b, g, r = cv2.split(mask)

new = cv2.bitwise_or(img, mask)

#new_img = np.dstack([blue, green, new])



cv2.imshow('img', new)
cv2.waitKey(0)
if cv2.waitKey(0) & 0xFF == ord('q'):
    sys.exit()

