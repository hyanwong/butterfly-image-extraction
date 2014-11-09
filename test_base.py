from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import urllib
import os
import csv
import re
import glob

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
    
save_contours=True

testcases = get_images(csv_file, image_dir)
for filenames in testcases:
    small_file = filenames[1]
    print("opening {} (ID={})".format(small_file,  tmpID))
    img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
    mask, params = best_circles(img)
    filename = os.path.splitext(small_file)[0]
    if save_contours:
        param_string = ''.join("%s=%r" % (key,val) for (key,val) in params.iteritems())
        cv2.imwrite(mask, os.path.join(os.path.dirname(filename),param_string, "_"+os.path.basename(filename)+".png"))
    else:
        #compare the current run with the saved files
        #read saved mask, of format blahblahblah_dID_***.png
        fileglob = os.path.join(os.path.dirname(filename), "*_"+os.path.basename(filename)+".png")
        saved_files = glob.glob(fileglob)
        if len(saved_files != 1):
            print("Couldn't find a single saved file matching {}".format(fileglob))
        else:
            target = cv2.imread(saved_files[0], cv2.IMREAD_GRAYSCALE)
            print(np.count_nonzero(np.logical_xor(target, mask)))