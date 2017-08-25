import cv2
import numpy as np

for i in range(1,7):
    piggy =cv2.imread("pig" + str(i)+".png")
    rpiggy = cv2.imread("pigr" +str(i) + ".png")
    piggy = cv2.cvtColor(piggy,cv2.COLOR_BGR2GRAY)
    rpiggy = cv2.cvtColor(rpiggy, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("g_pig" + str(i)+".png", piggy)
    cv2.imwrite("g_pigr" + str(i)+".png", rpiggy)
