import numpy as np
import cv2
import os
class pigs_template(object):
    def __init__(self):
        self.templates = []
        self.w = 68
        self.h = 47
        for i in [2, 5]:
            temp1= cv2.cvtColor(cv2.imread("images/pigs/pig" + str(i) + ".png"),cv2.COLOR_BGR2GRAY)
            temp2= cv2.cvtColor(cv2.imread("images/pigs/pig" + str(i) + "_rotate.png"),cv2.COLOR_BGR2GRAY)
            self.templates.append(temp1)
            self.templates.append(temp2)
            
    def getPigsLocation(self,res):
        w_pig = self.w
        h_pig = self.h
        points = []
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        while(max_val >=6003430):
            print(max_val)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            top_left = max_loc    
            bottom_right = (top_left[0] + w_pig, top_left[1] + h_pig)
            points.append((top_left, bottom_right))
            
            res[top_left[1]-15:bottom_right[1]+15, top_left[0]-15:bottom_right[0]+15] = -1.0
            _, n_max_val, _, n_max_loc = cv2.minMaxLoc(res)
            if(n_max_val == max_val):
                break
            else:
                max_val, max_loc = n_max_val,n_max_loc
        return points
    
    def getLocation(self,img):
        vecs = []
        for template in self.templates:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
            vecs = vecs + self.getPigsLocation(res)
        return vecs
