'''Call this script as "python circle_comparisons.py save" to save the outlines to an image_dir'''
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import urllib
import os
import csv
import re
import glob
import sys

from butterfly_detection import best_outline

#csv file in the same dir as this script, as dataID, URL
csv_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_butterfly.csv")
image_dir = "test_butterflies"

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
                filename = os.path.join(image_dir,dID+suffix)
                names.append(filename)
                if not(os.path.isfile(filename)):
                    print("getting {} from {}".format(filename, url))
                    urllib.urlretrieve(url, filename)
            filenames.append(names)
    return filenames
    

for filenames in get_images(csv_file, image_dir):
    large_file = filenames[0]
    small_file = filenames[1]
    dID = re.sub("_580_360.jpg$", "", os.path.basename(small_file))
    large_img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
    small_img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
    measure, params, mask = best_outline(small_img, large_img, dID, verbose=False)

    filename = os.path.splitext(small_file)[0]
    if sys.argv[1] == "save":
        param_string = '+'.join("%s=%s" % (key,val) for (key,val) in params.iteritems())
        maskfile = os.path.basename(filename)+"_"+param_string+".png"
        print("Writing best case file {}".format(maskfile))
        cv2.imwrite(os.path.join(os.path.dirname(filename),maskfile), mask)
    else:
        #compare the current run with the saved files
        #read saved mask, of format dID_blahblahblah_.png
        fileglob = os.path.join(os.path.dirname(filename), os.path.basename(filename)+"_*"+".png")
        saved_files = glob.glob(fileglob)
        if len(saved_files) > 1:
            print("Multiple matching saved outlines for {}".format(fileglob))
        elif len(saved_files) < 1:
            #this is not a pinned butterfly - we should assess how well we have detected this
            print("{}\t{}\t{}\t{}".format(large_file, large_img.shape[0:2], measure[0], measure[1]))
        else:
            target = cv2.imread(saved_files[0], cv2.IMREAD_GRAYSCALE)
            fit = np.count_nonzero(np.logical_xor(target, mask))/np.min(large_img.shape[0:2])
            print("{}\t{}\t{}\t{}\t{}".format(large_file, large_img.shape[0:2],measure[0], measure[1], fit))
