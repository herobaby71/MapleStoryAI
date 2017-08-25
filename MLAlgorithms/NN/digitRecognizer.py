import numpy as np
import cv2
import time
from FeedForwardNeuroNet import MultiLayerNeuroNet

digitRecognizer = None

def pad(img):
    w,h = img.shape
    pass
def getDigit(img, Model):
    x = np.reshape(img, (1, 784))
    
def main():
    digits = np.load("digits.npy")
    cv2.imwrite("forward_slash1.png",digits[3])
    cv2.imwrite("forward_slash2.png",digits[10])
    
##    arr = np.empty((len(digits), 784))
##    for i in range(len(digits)):
##        arr[i] = np.reshape(digits[i], (1,784))
##        cv2.imshow("digit", digits[i])
##        print(i)
##        if (cv2.waitKey(25) & 0xFF == ord('q')):
##            cv2.destroyAllWindows()
##            break
##        time.sleep(6)
   # np.save("digits_finale.npy", new_digits)
main()

