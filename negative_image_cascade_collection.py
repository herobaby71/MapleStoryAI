import urllib.request
import numpy as np
import cv2
import os
from getScreen import grab_screen
from MapleStoryControl import Control
import time
#pig 68x40, scale down to 34x20

count = 1
pos_file_location = "haar-cascade/pos"
neg_file_location = "haar-cascade/neg"
def collect_from_game():
    global count
    if not os.path.exists(pos_file_location):
        os.makedirs(pos_file_location)
        
    while(True):
        print(count)
        file_name = pos_file_location + "/" +str(count) + ".jpg"
        screen =  cv2.cvtColor(grab_screen(region = (0,100,800,540)),cv2.COLOR_BGR2GRAY)
        mini_screen = cv2.resize(screen, (400,220))
        cv2.imshow("mini_screen", mini_screen)

        cv2.imwrite(file_name, mini_screen)
        count+=1
        if(cv2.waitKey(25) & 0xFF == 27):
            cv2.destroyAllWindows()
            break
        time.sleep(1.5)

#pythonprogramming.net opencv tutorials
def collect_from_image_net():
    global count
    
    neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09411189'   
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
          
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            if(count > 15):
                file_name = "haar-cascade/background/" +str(count) + ".jpg"
                urllib.request.urlretrieve(i, file_name)
                img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
                # should be larger than samples / pos pic (so we can place our image on it)
                resized_image = cv2.resize(img, (100, 100))
                cv2.imwrite(file_name,resized_image)
            count+=1
            
        except Exception as e:
            print(str(e))
#pythonprogramming.net opencv tutorials
def find_uglies():
    match = False
    for file_type in [pos_file_location]:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def create_pos_n_neg():
    for file_type in [neg_file_location]:
        for img in os.listdir(file_type):
            if file_type == neg_file_location:
                line = file_type+'/'+img+'\n\n'
                with open('bg.txt','a') as f:
                    f.write(line)
create_pos_n_neg()
