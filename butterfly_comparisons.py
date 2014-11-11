'''Call this script as "python butterfly_comparisons.py" to download (if necessary) and run comparisons on all the images in "test_butterfly.csv"
   call this script as "python butterfly_comparisons.py save" to save the best outlines to an image_dir, or
   call this script as "python butterfly_comparisons.py imagefile_orig.jpg" to display to screen the best outline and a tiled image of the fitting steps'''

#NB, this might be useful: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

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
try:
    from statsmodels.formula.api import logit
    import pandas as pd
    do_logistic_regression = True
except ImportError:
    do_logistic_regression = False

from butterfly_detection import best_outline

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


def show_working(small_img, large_img, dID):
    measure, params, mask = best_outline(small_img, large_img, dID, composite_file_dir="", verbose=True)
    contours = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    x,y,w,h = cv2.boundingRect(contours[0])
    cv2.imshow("Full res mask", mask[y:y+h,x:x+w])
    cv2.waitKey()



if len(sys.argv) > 1 and re.search("\\.jpg$", sys.argv[1]):
    for file in (sys.argv[1:]):
        small_file = cv2.imread(re.sub("_\w+\.jpg$", "_580_360.jpg", str(file)), cv2.CV_LOAD_IMAGE_COLOR)
        large_file = cv2.imread(re.sub("_\w+\.jpg$", "_orig.jpg", str(file)), cv2.CV_LOAD_IMAGE_COLOR)
        dID = re.sub("_.*$", "", str(file))
        show_working(small_file, large_file, dID)

else:
    #csv file in the same dir as this script, as dataID, URL
    csv_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_butterfly.csv")
    image_dir = "test_butterflies"
    
    
    stats = []
    for filenames in reversed(get_images(csv_file, image_dir)):
        large_file = filenames[0]
        small_file = filenames[1]
        dID = re.sub("_580_360.jpg$", "", os.path.basename(small_file))
        large_img = cv2.imread(large_file, cv2.CV_LOAD_IMAGE_COLOR)
        small_img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
        measure, params, mask = best_outline(small_img, large_img, dID, verbose=False)
    
        filename = os.path.splitext(small_file)[0]
        if len(sys.argv) > 1 and sys.argv[1] == "save":
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
            else:
                if len(saved_files) < 1:
                    #this is not a pinned butterfly - we should assess how well we have detected this
                    fit = np.nan
                    print("{}\t{}\t{}\t{}".format(large_file, large_img.shape[0:2], measure[0], measure[1]))
                else:
                    target = cv2.imread(saved_files[0], cv2.IMREAD_GRAYSCALE)
                    fit = np.count_nonzero(np.logical_xor(target, mask))/np.min(large_img.shape[0:2])
                    print("{}\t{}\t{}\t{}\t{}".format(large_file, large_img.shape[0:2],measure[0], measure[1], fit))
                stats.append([fit, measure[0], measure[1]])
               
    if len(stats):
        stats = np.asarray(stats)
        masked_stats = np.ma.masked_array(stats,np.isnan(stats))
    
        if do_logistic_regression:
            dat = pd.DataFrame({"Butterfly": np.where(np.isnan(stats[:,0]), 0, 1), "pr_but":stats[:,1], "floodfill_percent":stats[:,2]})
            logit_model = logit(formula = 'Butterfly ~ pr_but + floodfill_percent', data = dat).fit()
            print(logit_model.summary())
        print("Av. mask disparity: {}".format(np.mean(masked_stats[:,0])))

