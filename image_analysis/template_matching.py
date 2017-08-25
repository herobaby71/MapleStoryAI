import numpy as np
import cv2
import time
from getScreen import grab_screen
from pigs_template import pigs_template

portal_template = cv2.cvtColor(cv2.imread("images/earth.png"),cv2.COLOR_BGR2GRAY)
pig_template = cv2.cvtColor(cv2.imread("images/pigs/pig-template.png"),cv2.COLOR_BGR2GRAY)
character_template = cv2.cvtColor(cv2.imread("images/character/characther_template.png"),cv2.COLOR_BGR2GRAY)
origin_template = cv2.cvtColor(cv2.imread("images/origin.png"),cv2.COLOR_BGR2GRAY)

def getLocation(image, template):
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _,_,_,top_left = cv2.minMaxLoc(res)
    bottom_right = (top_left[0] + w,top_left[1] + h)
    return (top_left, bottom_right)

#make sure that the region sur
def checkAndGetBoundary(region, left, right, top, bottom):
    h,w = region.shape[0],region.shape[1]
    if(left <= 0):
        left = 0
    if(right >= w):
        right = w
    if(top <= 0):
        top = 0
    if(bottom >=h):
        bottom = h
    return (left, right, top, bottom)

def getLocations(image, template, max_count = 10, thresh = 7093430):
    points = []
    w, h = template.shape[1], template.shape[0]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    count = 1
    while(max_val >=7093430 and count <=max_count):
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        points.append((top_left, bottom_right))

        #since minMaxLoc return the global peak, we want to get rid of it so that we can
        #get a new global peak
        left, right, top, bottom = checkAndGetBoundary(res, top_left[0]-10, bottom_right[0]+10,top_left[1]-7, bottom_right[1]+7) 
        res[top:bottom, left:right] = -100
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        count+=1
    return points
def reducedimensions(image, lane_locs, character_locs, mob_locs):
    """
        reduce the dimension of the image to 20 by 11 matrix with locations
        of monsters, lanes and characters
    """
    
def generateMap(file_location = "images/maps/Henesys_Pig_Farm.txt"):
    with open(file_location) as f:
        lines = f.readlines()
    map_locations = []
    for line in lines:
        map_locations.append(list(map(int,line.strip().split())))
        
    while (True):
        time_start = time.time()
        img =  grab_screen(region = (0,100,800,540))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #the portal seems to be a problem so any detections surrounding it is unvalid
        ori_top_left, ori_bottom_right = getLocation(gray, origin_template)

##        origin = ((ori_top_left[0]+ ori_bottom_right[0])/2,(ori_top_left[1]+ ori_bottom_right[1])/2)
        origin = ori_top_left
        #draw map
        for line in map_locations:
            p1 = tuple(map(int,map(sum, zip((line[0], line[1]), origin))))
            p2 = tuple(map(int,map(sum, zip((line[2],line[3]), p1))))
            cv2.rectangle(img, p1, p2, (0,255,255),2)
        
        cv2.rectangle(img,ori_top_left,ori_bottom_right, (255,0,0),2)
        cv2.imshow("screen", img)
        
        if(cv2.waitKey(25) & 0xFF == 27):
            cv2.destroyAllWindows()
            break
        time_end = time.time()
        print("total loop took {}".format(str(time_start-time_end)))
    file.close()
def main():
    while (True):
        time_start = time.time()
        img =  grab_screen(region = (0,100,800,540))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #the portal seems to be a problem so any detections surrounding it is unvalid
        port_top_left, port_bottom_right = getLocation(gray, portal_template)
        #find main character by getting the top left location and bottom right location
        char_top_left,char_bottom_right = getLocation(gray, character_template)
        
        #apply template matching to find piggies
        pigVecs = getLocations(gray, portal_template, 10,15093430)

        newVecs = []
        #draw rectangles 
        for i,vec in enumerate(pigVecs):
            if((abs(vec[0][0]-char_top_left[0]) < 15) and (abs(vec[0][1]-char_top_left[1])<15)):
                print("yordle")
                continue
            newVecs.append(vec)
            cv2.rectangle(img,vec[0], vec[1], (0,255,0), 2)
        
        cv2.rectangle(img,char_top_left,char_bottom_right, (0,255,255),2)
        
        cv2.rectangle(img,port_top_left,port_bottom_right, (255,0,0),2)
        
        cv2.imshow("screen", img)        
        if(cv2.waitKey(25) & 0xFF == 27):
            cv2.destroyAllWindows()
            break
        time_end = time.time()
        print("total loop took {}".format(str(time_start-time_end)))
    
generateMap()
