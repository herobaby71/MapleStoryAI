import numpy as np
import cv2
import time
from image_analysis.getScreen import grab_screen
from KeyInputs.MapleStoryControl import Control
from MLAlgorithms.NN.FeedForwardNeuroNet import MultiLayerNeuroNet


def main():
    Thetas = np.load("Thetas1.npy")
    DataX = np.load("DataX.npy")
    DataY = np.load("DataY.npy")
    model = MultiLayerNeuroNet(DataX,DataY,1080, 1080, 10, 0, Thetas,"ReLU", "softmax")
    control = Control()
    commands = {0:"left",1:"right", 2:"climb", 3:"jump", 4:"leftjump", 5:"rightjump", 6:"leftjumpclimb", 7:"rightjumpclimb", 8:"downjump", 9:"attack"}
    while(True):
        screen =  grab_screen(region = (0,100,800,540))
        mini_screen = cv2.resize(screen, (40,27))
        cv2.namedWindow("mini_screen",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mini_screen", 200,100)
        cv2.imshow("mini_screen", mini_screen)
        if(cv2.waitKey(25) & 0xFF == 27):
            cv2.destroyAllWindows()
            break
##        x = np.reshape(mini_screen, (1, 1080))
        #y = model.predict(x)
        #control.command(commands[y[0]])
if __name__ == "__main__":
    main()
