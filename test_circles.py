'''Call e.g. as python test_circles.py filename_580_360.jpg, or without any parameters to search the images_eol dir for such files'''
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import re
import sys

from circle_detection import best_outline


def test_circle(filename):
    dID = re.sub("_580_360.jpg$", "", os.path.basename(filename))
    print("opening {} (ID={})".format(filename,  dID))
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_COLOR)
    measures, params, mask = best_outline(img, True)
    #find circles in the accumulated mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    display_image=cv2.merge((mask,mask,mask)) ## Create a 3-channel display image
    for c in contours:
        (cx,cy),cr = cv2.minEnclosingCircle(c)
        print("{} {} {}".format(cx,cy,cr)) #ADDED PARENTHESES
        cv2.circle(display_image, (int(cx),int(cy)),int(cr*.75), color=(0,0,255), thickness=cv2.cv.CV_FILLED) ## Add red circles
    cv2.imshow("found circles", display_image) #ADDED COMMA
    cv2.waitKey()

    print(params)




if len(sys.argv) > 1:
    for file in (sys.argv[1:]):
        test_circle(str(file))
else:
    image_folder = 'images_eol' #this should contain ID_580_360.jpg files.
    if os.path.isdir(image_folder):
        pattern = re.compile("(.*)_580_360.jpg$"); #only opens small images
        for img_file in os.listdir(image_folder):
            match = pattern.search(img_file)
            if match is not None:
                tmpID = match.group(1)
                small_file = os.path.join(image_folder, img_file)
                test_circle(small_file)
            else:
                print("not a _580_360.jpg file")                

