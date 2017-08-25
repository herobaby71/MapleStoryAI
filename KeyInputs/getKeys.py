
import pywintypes
import win32api, win32con
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if (win32api.GetAsyncKeyState(ord(key))):
            keys.append(key)
    if(win32api.GetAsyncKeyState(win32con.VK_UP)):
       keys.append("UP")
    if(win32api.GetAsyncKeyState(win32con.VK_DOWN)):
       keys.append("DOWN")
    if(win32api.GetAsyncKeyState(win32con.VK_LEFT)):
       keys.append("LEFT")
    if(win32api.GetAsyncKeyState(win32con.VK_RIGHT)):
       keys.append("RIGHT")
    return keys

##while(True):
##    keys = key_check()
##    if("UP" in keys):
##        print("yolo")
##    elif("DOWN" in keys):
##        print("molo")
