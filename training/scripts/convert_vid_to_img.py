import cv2
import os

# Delcare where to save the pictures
saving_path = 'C:/Users/user1/drone_video_pics'  # Change as needed

# Open the video file
cap = cv2.VideoCapture('C:/Users/user1/drone_vid.mp4')  # Change as needed

# Loop through the video
file_number = 0
while(cap.isOpened()):

    # Get a frame
    ret,frame = cap.read()
    if ret == False:
        break
    # Save the frame to the path
    file_name = f'drone_{file_number}.jpg'
    cv2.imwrite(os.path.join(saving_path, file_name), frame)
    file_number += 1

    print(f'Saved file: {file_name}')

    # Skip 5 frames. First number is inclusive, second number is exclusive
    for x in range(1,6):
        ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()