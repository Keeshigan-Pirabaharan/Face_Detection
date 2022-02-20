import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture Frame by Frame
    ret, frame = cap.read()
    # Convert to grayscale for haar to work
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # x/y is bottom of face frame and w and h adds to opposite corner
    for(x, y, w, h) in faces:
        print(x,y,w,h)
        # roi = region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Recognize? can use deep learning model (tensorflow) to learn

        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        # Draw a rectangle
        color = (255, 0, 0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0XFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()