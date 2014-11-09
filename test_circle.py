from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import re
import sys

from circle_detection import best_outline

for file in (sys.argv[1:]):
    small_file = str(file)
    dID = re.sub("_580_360.jpg$", "", os.path.basename(small_file))

    print("opening {} (ID={})".format(small_file,  dID))
    img = cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR)
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
