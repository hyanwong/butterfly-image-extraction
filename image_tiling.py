from __future__ import division
import cv2
import numpy as np

class tiled_img:
    '''make an image consisting of a number of pictures tiled together. By default add left to right. Use add(..., newrow=TRUE) to add a new row.'''
    def __init__(self):
        self.main_image = None
        self.prev_right_edge = 0
        self.colours = [[255,0,0],[0,0,255], [0,255,0],[0,255,255],[255,0,255],[255,255,0]]

    def add(self, img, name=None, newrow=False, contours=None, focal_contours=[]):
        #first convert to BGRA with alpha channel, so we can overlay translucent contours
        if len(img.shape) < 3 or img.shape[2] == 1:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGRA)
        elif len(img.shape) < 4:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        else:
            img = img.copy()
            
        h, w = img.shape[:2]
    
        if self.main_image is None:
            self.main_image = np.zeros((h,w,3), np.uint8)

        if newrow:
            newrow = np.zeros((h,self.main_image.shape[1],3), np.uint8)
            self.main_image = np.concatenate((self.main_image,newrow)) #add enough space for another row
            self.prev_right_edge = 0

        width_diff = self.prev_right_edge + w - self.main_image.shape[1]
        if width_diff > 0:
            self.main_image = np.concatenate((self.main_image, np.zeros((self.main_image.shape[0],width_diff,3), np.uint8)), 1) #extend main.image width to allow for new pic
    
        if contours is not None:
            cv2.drawContours(img, contours, -1, self.colours[0] + [100])
            for i in range(len(focal_contours)):
                cv2.drawContours(img, contours, focal_contours[i], self.colours[i+1] + [200])
            
        if name is not None:
            cv2.putText(img, name, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)

        bottom = self.main_image.shape[0]
        self.main_image[(bottom-h):bottom, self.prev_right_edge:self.prev_right_edge+w] = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        self.prev_right_edge += w

    def imwrite(self, filename):
        cv2.imwrite(filename, self.main_image)

    def imshow(self, name):
        cv2.imshow(name, self.main_image)
