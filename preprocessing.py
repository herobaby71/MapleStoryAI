import numpy as np
import cv2
import time
from getScreen import grab_screen
from FeedForwardNeuroNet import MultiLayerNeuroNet
from CompareImages import CompareImage
from imutils import contours
#locations on the screen
HEALTH = (242,495, 288, 505)
MANA = [356,495,400,505]
EXP = [469,495,499,505]

#Model for digit recognizer
Thetas = np.load("MapleDigitThetas.npy")
model = MultiLayerNeuroNet(None, None, 784, 784, 12, .5, Thetas,"ReLU", "softmax")

#load all digits image in the Digits directory to compare
CompareImg = CompareImage()

#pad all sides given dim
def pad(img, dim):
    top = np.zeros((dim[0][0], img.shape[1]))
    bot = np.zeros((dim[0][1], img.shape[1]))
    img = np.concatenate((top, img), axis = 0)
    img = np.concatenate((img, bot), axis = 0)
    left = np.zeros((img.shape[0],dim[1][0]))
    right = np.zeros((img.shape[0],dim[1][1]))
    img = np.concatenate((left, img), axis = 1)
    img = np.concatenate((img, right), axis = 1)
    return img

def padding(img):
    pad_dims = [[0,0],[0,0]]
    m,n = img.shape
    if((28-m)%2 == 0):
        pad_dims[0][0] = int((28-m)/2)
        pad_dims[0][1] = int((28-m)/2)
    else:
        pad_dims[0][0] = int((28-m)/2)
        pad_dims[0][1] = int((28-m)/2)+1
    if((28-n)%2 == 0):
        pad_dims[1][0] = int((28-n)/2)
        pad_dims[1][1] = int((28-n)/2)
    else:
        pad_dims[1][0] = int((28-n)/2)
        pad_dims[1][1] = int((28-n)/2)+1
    #return pad(img, pad_dims)
    return np.pad(img, pad_dims, 'constant', constant_values=(0,0))

#take an image and return the number in the image
#recognize using histogram comparisions
def recognizeDigits(img):
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    thresh = cv2.resize(thresh, (thresh.shape[1]*3,thresh.shape[0]*3))
    _,cnts,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = []
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        if(h > 18 and w > 6):
            digitCnts.append(c)

    #sort the digits from left to right
    try:
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    except: pass
    numbers = ""
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        roi = thresh[y:y+h,x:x+w]
        pad = padding(roi)
        numbers+=CompareImg.compare_to_all_digits(pad)
    return numbers

#recognize using neuronetworks
#doesnt work well
def recognizeDigitsNN(img):
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    thresh = cv2.resize(thresh, (thresh.shape[1]*3,thresh.shape[0]*3))
    _,cnts,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = []
    digitCnts = []
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        if(h > 18 and w > 6):
            digitCnts.append(c)
    try:
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    except: pass

    numbers = ""
    for c in digitCnts:
        (x,y,w,h) = cv2.boundingRect(c)
        roi = thresh[y:y+h,x:x+w]
        pad = padding(roi)
        x = np.reshape(pad, (1,784))
        y = model.predict(x/255)[0]
        if(y == 10): numbers+="["
        elif(y == 11): numbers+="/"
        else: numbers+=str(y)
    return numbers
            
def HEALTH_ROI(image):
    return cv2.cvtColor(image[HEALTH[1]: HEALTH[3], HEALTH[0]:HEALTH[2]],cv2.COLOR_BGR2GRAY)

def MANA_ROI(image):
    return cv2.cvtColor(image[MANA[1]: MANA[3], MANA[0]:MANA[2]],cv2.COLOR_BGR2GRAY)

def EXP_ROI(image):
    return cv2.cvtColor(image[EXP[1]: EXP[3], EXP[0]:EXP[2]],cv2.COLOR_BGR2GRAY)

#name can be "health", "mana", or "exp"
def getValue(name = "health", screen = None):
    ans = ""
    if (name == "health"):
        ans = recognizeDigits(HEALTH_ROI(screen))
    elif(name == "mana"):
        ans = recognizeDigits(MANA_ROI(screen))
    elif(name == "exp"):
        ans = recognizeDigits(EXP_ROI(screen))
    return ans
def main():
    while(True):
        start = time.time()
        screen =  grab_screen(region = (0,100,800,610))
##        health = getValue("health", screen)
##        print("health:" + health)
##        exp = getValue("exp", screen)
        mana = getValue("mana", screen)
        print("mana:" + mana)
        #print("exp:" + exp)
        #cv2.imshow("original", screen)
        
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
        end = time.time()
        print("loop took:{}".format(str(end-start)))
if __name__ == "__main__":
    main()
