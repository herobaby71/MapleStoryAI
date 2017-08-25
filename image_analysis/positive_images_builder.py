import cv2
import os

#width and height of the object. reseted after the first selection
obj_dims =[]

def select_objects(img):
    """
    Select a region of the image by pressing and dragging the mouse
    Return the regions in the form of vectors [(x,y,w,h),...]
    """
    nimg = img.copy()
    obj_vectors = []
    roi_pts = [] #region selected: define by first and last points
    def select_event(event, x, y, flags, params):
        nonlocal roi_pts
        nonlocal obj_vectors
        
        if (event == cv2.EVENT_LBUTTONDOWN):
            roi_pts = [(x,y)]
        if (event == cv2.EVENT_LBUTTONUP):
            roi_pts.append((x,y))

            #get x,y,width and heights
            x = roi_pts[0][0]
            y = roi_pts[0][1]
            w = roi_pts[1][0] - roi_pts[0][0]
            h = roi_pts[1][1] - roi_pts[0][1]
            obj_dims = [w,h]
            obj_vectors.append((x, y, w,h))

            #draw rect around that area
            cv2.rectangle(nimg, roi_pts[0], roi_pts[1], (0,255,255),2)
            cv2.imshow("image", nimg)
            
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_event)

    while(True):
        cv2.imshow("image",nimg)
        if (cv2.waitKey(25) & 0xFF == 27):
            #confirm selection
            cv2.destroyAllWindows()
            break
        elif(cv2.waitKey(25) & 0xFF == ord('z')):
            #redo
            nimg = img.copy()
            try:
                del obj_vectors[-1]
            except(Exception):
                pass
            for vec in obj_vectors:
                cv2.rectangle(nimg, (vec[0],vec[1]), (vec[0]+vec[2],vec[1]+vec[3]), (0,255,255),2)
    return obj_vectors


file = open('info.txt', 'a')
def create_positives(folder_location):
    #open file
    global file
    reachSavePoint = False
    for file_type in [folder_location]:
        for img in os.listdir(file_type):
            if(img == "1103.jpg"):
                reachSavePoint = True
            if(not(reachSavePoint)):
                continue
            current_image_path =  str(file_type) + "/" + str(img)
            image = cv2.imread(current_image_path)
            vecs = select_objects(image)
            try:
                if(len(vecs) == 0):
                    print("file: " + current_image_path + " is not a positive image")
                    print("move to neg")
                    new_image_path = "neg/" + "aaaaaa" +str(img) 
                    os.rename(current_image_path, new_image_path)
                    continue
                #append the information to the file
                line = current_image_path +" " +str(len(vecs))
                for vec in vecs:
                    line +=" " + str(vec[0]) + " " +  str(vec[1]) + " " + str(vec[2]) + " " + str(vec[3])
                line+= "\n"
                file.write(line)
            except Exception:
                pass
    file.close()
def main():
    folder_location = "pos"
    if not os.path.exists(folder_location):
        print("path not exists")
    else:        
        create_positives(folder_location)

main()
file.close()
##img = cv2.imread("pos/test.jpg")
##vecs = select_objects(img)
##print(vecs)
##for vec in vecs:
##    cv2.rectangle(img, (vec[0],vec[1]), (vec[0]+vec[2],vec[1]+vec[3]), (0,255,255),2)
##cv2.imshow("image", img)
