import numpy as np
import cv2
import time
from getScreen import grab_screen

pig_cascade = cv2.CascadeClassifier('haar-cascade/mage-character-cascade.xml')
##img =  cv2.resize(grab_screen(region = (0,100,800,540)),(400,220))
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##pigs = pig_cascade.detectMultiScale(gray,30,40)
##for (x,y,w,h) in pigs:
##    print(x,y,w,h)
##    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,255),2)
##    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,255),2)
##cv2.imshow("img",img)
##cap = cv2.VideoCapture(0)
count = 0
while True:
    time_start = time.time()
    img =  grab_screen(region = (0,100,800,540))
    small_img = cv2.resize(img, (200,110))
    small_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    time_start = time.time()
    pigs = pig_cascade.detectMultiScale(small_gray, scaleFactor =100, minNeighbors = 600)  #draw a box around it
    time_end = time.time()
    for (x,y,w,h) in pigs:
        cv2.rectangle(small_img,(x,y),(x+w,y+h), (0,255,255),2)
    cv2.imshow('img', small_img)
    if(cv2.waitKey(25) & 0xFF == 27):
        break
    time_end = time.time()
    print("total loop took {}".format(str(time_start-time_end)))
    
##cv2.destroyAllWindows()
