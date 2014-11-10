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

from circle_detection import best_outline

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

def show_working(filename):
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and re.search("\\.jpg$", sys.argv[1]):
        for file in (sys.argv[1:]):
            show_working(str(file))

    else:
        #csv file in the same dir as this script, as dataID, URL
        csv_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "circles.csv")
        image_dir = "test_circles"
        
        for filenames in get_images(csv_file, image_dir):
            small_file = filenames[1]
            dID = re.sub("_580_360.jpg$", "", os.path.basename(small_file))
            img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
            contours, params, mask = best_outline(img)
            circles=0
            for c in contours:
                if cv2.contourArea(c) < 100:
                    print("small circle in {} not counted".format(small_file))
                else:
                    circles +=1;
        
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
                if len(saved_files) != 1:
                    print("Couldn't find a single saved file matching {}".format(fileglob))
                else:
                    target = cv2.imread(saved_files[0], cv2.IMREAD_GRAYSCALE)
                    contours = cv2.findContours(target.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
                    target_circles=0
                    for c in contours:
                        if cv2.contourArea(c) < 100:
                            print("small circle in {} not counted".format(fileglob))
                        else:
                            target_circles += 1;
                    print("{}\t{}\t{}".format(small_file,np.count_nonzero(np.logical_xor(target, mask)), circles-target_circles))
