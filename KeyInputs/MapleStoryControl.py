import numpy as np
import cv2
import time
from directKeyInputs import PressKey,ReleaseKey, W, A, S, D, X, Z, SKILL1, UP, DOWN, LEFT, RIGHT

#Labels name
#  0     1   2   3   4     5      6        7        8      9
#[Left,Right,Up, X,LeftX,RightX,LeftUpX,RightUpX, DownX,Skill1]
class Control:
    def __init__(self):
        self.keys = (X,SKILL1, UP,DOWN, LEFT, RIGHT)
    def command(self, name):
        getattr(self, name)()
    def releaseAllKey(self):
        for i in self.keys:
            ReleaseKey(i)
    def left(self):
        self.releaseAllKey()
        PressKey(LEFT)
    def right(self):
        self.releaseAllKey()
        PressKey(RIGHT)
    def climb(self):
        self.releaseAllKey()
        PressKey(UP)
    def jump(self):
        PressKey(X)
    def leftjump(self):
        self.releaseAllKey()
        PressKey(X)
        PressKey(LEFT)
    def rightjump(self):
        self.releaseAllKey()
        PressKey(RIGHT)    
        PressKey(X)   
    def leftjumpclimb(self):
        self.releaseAllKey()
        PressKey(LEFT)       
        PressKey(X)
        PressKey(UP)
    def rightjumpclimb(self):
        self.releaseAllKey()
        PressKey(RIGHT)
        PressKey(X)
        PressKey(UP) 
    def downjump(self):
        self.releaseAllKey()
        PressKey(DOWN)
        PressKey(X)
    def attack(self):
        PressKey(SKILL1)
        time.sleep(.25)
        ReleaseKey(SKILL1)
if __name__ == "__main__":
    temp = Control()
    time.sleep(5)
    temp.right()
    time.sleep(3)
    temp.left()
    time.sleep(3)
    temp.jump()
    time.sleep(2)
    ReleaseKey(X)
    
