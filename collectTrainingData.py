import cv2
import numpy as np
import time
import os
from getScreen import grab_screen
from getKeys import key_check


train_data = []
set_count = 1
file_name = 'training_data1.npy'
while(os.path.isfile(file_name)): 
    file_name = "training_data" + str(set_count) + ".npy"
    set_count+=1
print("file name:" + file_name)
##if os.path.isfile(file_name):
##    print("file exists, opening file...")
##    train_data = list(np.load(file_name))
##else: print("file does not exists, create new")

#grabing keys from maple
def keys_to_array(keys):
    """
        convert keys into array for multi classification purposes
          0     1   2   3   4     5     6         7        8      9
        [Left,Right,Up, X,LeftX,RightX,LeftUpX,RightUpX, DownX,Skill1]
    """
    ans = [0,0,0,0,0,0,0,0,0,0]

    if("LEFT" in keys and "X" in keys):
        if("UP" in keys): ans[6] = 1
        else: ans[4] = 1
    elif("RIGHT" in keys and "X" in keys):
        if("UP" in keys): ans[7] = 1
        else: ans[5] = 1
    elif("DOWN" in keys and "X" in keys):
        ans[8] = 1
    elif("1" in keys):
        ans[9] = 1
    elif("X" in keys):
        ans[3] = 1
    elif("UP" in keys):
        ans[2] = 1
    elif("LEFT" in keys):
        ans[0] = 1
    elif("RIGHT" in keys):
        ans[1] = 1
    else:
        ans = None
    return ans

#count_down until begin collecting data
for i in range(5):
    print("",5-i)
    time.sleep(1)

#collect data
def main(file_name, set_count):
    count = 0
    start_collecting = False 
    while (True):
        #start = time.time()
        screen = cv2.cvtColor(grab_screen(region = (0,100,800,540)),cv2.COLOR_BGR2GRAY)
        mini_screen = cv2.resize(screen, (160,108))
        
        cv2.imshow('screen',screen)
        
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

        keys = key_check()
        label = keys_to_array(keys)
        
        if('P' in keys):
            start_collecting = not(start_collecting)
            if(start_collecting):
                print("unpaused")
            else: print("paused")
            
        if(start_collecting and label is not None):
            train_data.append([mini_screen, label])
            count+=1

        if(count %500 == 0):
            print("have collected {} data points".format(str(count)))
        #save after every 5000 datapoints
        if(count%20000 == 0):
            print("have collected {} data points".format(str(count)))
            np.save(file_name, train_data)
            if(count == 40000):
                set_count+=1
                file_name = "training_data" + str(set_count) + ".npy"
##        end = time.time()
##        print("Loop takespp {}".format(str(end-start)))
        
main(file_name, set_count)
np.save(file_name, train_data)
