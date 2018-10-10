"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from:
http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

Use this code as a template to process images in real time, using the same techniques as the last challenge.
You need to display a gray scale video with 320x240 dimensions, with box at the center
"""


# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (224,224)
camera.framerate = 32
#camera.color_effects = (128, 128)
rawCapture = PiRGBArray(camera, size=(224,224))

# allow the camera to warmup
time.sleep(0.1)
i = 25 

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(gray, (130,10) ,(30,110), (255,255,255))
 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),2)
           
            if key == ord("p"):
                #camera.capture('/home/pi/myFaceRecognition/images/17/image%s.jpg' % i)
                #imgcrop = cv2.imread('/home/pi/myFaceRecognition/images/17/image%s.jpg' % i)
                #i += 1
                crop_img = gray[y:y+h, x:x+w]
                cv2.imwrite( '/home/pi/myFaceRecognition/images/17/image%s.jpg' % i, crop_img)
                i += 1

    # show the frame
    cv2.imshow("Frame", gray)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
