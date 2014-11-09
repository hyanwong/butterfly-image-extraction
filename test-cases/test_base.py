from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import urllib
import os
import csv
import re
from circle_detection import best_circles


#csv file in the same dir as this script, as dataID, URL
csv_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "circles.csv")
image_dir = "test_circles"

def get_images(csv_file, image_dir):
    filenames = []
    with open(csv_file, 'r') as f:
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)  
        reader = csv.reader(f)
        for row in reader:
            dID = re.sub("[^0-9]", "", row[0])
            names = []
            for suffix in ("_orig.jpg", "_580_360.jpg"):
                url = re.sub("_\w+\.jpg$", suffix, row[1])
                filename = dID+suffix
                names.append(filename)
                if not(os.path.isfile(filename)):
                    print("getting {} from {}".format(os.path.join(image_dir, filename), url))
                    urllib.urlretrieve(row[1], os.path.join(image_dir, filename), url)
            filenames.append[names]
    return filenames
    
def save_outlines():
    pass    
    
if 
testcases = get_images(csv_file, image_dir)
for filenames in testcases:
    small_file = filenames[1]
    print("opening {} (ID={})".format(small_file,  tmpID))
    img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
    if save_contours:
        mask, params = best_circles(img)
        param_string = ''.join("%s=%r" % (key,val) for (key,val) in params.iteritems())
        cv2.imwrite(mask, param_string+small_file)
    else:
        #compare the current run with the saved files