import numpy as np
import cv2

class CompareImage(object):

    def __init__(self):
        self.minimum_commutative_image_diff = .01
        digits = []
        for i in range(0,10):
            digits.append(cv2.imread("Digits/digit_{}.png".format(str(i)),0))
        digits.append(cv2.imread("Digits/digit_[.png",0))
        digits.append(cv2.imread("Digits/digit_slash.png", 0))
        self.digits = digits
    
    def compare_images(self, img1, img2):
        image_1 = img1
        image_2 = img2
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            return commutative_image_diff
        return 10000 #random failure value
    
    def compare_to_all_digits(self,img):
        commutative_image_diffs = [0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(0,len(commutative_image_diffs)):
            commutative_image_diffs[i] = self.get_image_difference(img, self.digits[i])
        min_index = np.argmin(commutative_image_diffs)
        
        if min_index == 10: return "["
        elif min_index == 11: return "/"
        return str(min_index)
    
    #author: Priyanshu Chauhan
    #Stack Exchange: https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff

##img = cv2.imread("Digits/digit_slash.png")
##c = CompareImage()
##arr =c.compare_to_all_digits(img)
