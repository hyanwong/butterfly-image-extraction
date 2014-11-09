from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import urllib
import os
import csv
import re
#csv file in the same dir as this script, as dataID, URL
csv_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "circles.csv")
image_dir = "test_circles"

def get_image_files(csv_file, image_dir):
    with open(csv_file, 'r') as f:
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)  
        reader = csv.reader(f)
        for row in reader:
            dID = re.sub("[^0-9]", "", row[0])
            for suffix in ("_orig.jpg", "_580_360.jpg"):
                url = re.sub("_\w+\.jpg$", suffix, row[1])
                filename = dID+suffix
                if not(os.path.isfile(filename)):
                    print("getting {} from {}".format(os.path.join(image_dir, filename), url))
                    urllib.urlretrieve(row[1], os.path.join(image_dir, filename), url)
        
def save_outlines():
    pass    
    
def compare(function):
   
   
   
    pattern = re.compile("(.*)_580_360.jpg$"); #only opens small images
    for img_file in os.listdir(image_folder):
        match = pattern.search(img_file)
        if match is not None:
            tmpID = match.group(1)
            small_file = os.path.join(image_folder, img_file)
            print("opening {} (ID={})".format(small_file,  tmpID))
 
 
get_image_files(csv_file, image_dir)

    print("opening {} (ID={})".format(small_file,  tmpID))
    img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
