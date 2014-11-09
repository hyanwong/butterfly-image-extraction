from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import re

from circle_detection import best_circles

image_folder = 'images_eol' #this should contain ID_580_360.jpg files.
if os.path.isdir(image_folder):
    pattern = re.compile("(.*)_580_360.jpg$"); #only opens small images
    for img_file in os.listdir(image_folder):
        match = pattern.search(img_file)
        if match is not None:
            tmpID = match.group(1)
            small_file = os.path.join(image_folder, img_file)
            print("opening {} (ID={})".format(small_file,  tmpID))
            img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
            best_circles(img, True)
        else:
            print("not a _580_360.jpg file")                



