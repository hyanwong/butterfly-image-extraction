from __future__ import division
from __future__ import print_function
import cv2
import numpy as np


def remove_shadows(img):
    '''Not yet implemented. To start, see A Statistical Approach for Real-time Robust Background Subtraction and Shadow Detection
     http://vast.uccs.edu/~tboult/frame/Horprasert/HorprasertFRAME99.pdf
     or http://www.serc.iisc.ernet.in/~venky/SE263/papers/Salvador_CVIU2004.pdf
    
     c1c2c3 colour space is very useful. Can use all 3 channels by detecting edges in each independently, then merging edges together
    '''
    
    #look for symmetry before 
    
    ##
    ## Some alternative options, using c1c2c3 colour space (also useful for deshadowing)
    ##

    im = img.astype(np.float32)+0.001 #to avoid division by 0
    c1c2c3 = np.arctan(im/np.dstack((cv2.max(im[...,1], im[...,2]), cv2.max(im[...,0], im[...,2]), cv2.max(im[...,0], im[...,1]))))
    c1c2c3 = cv2.normalize(c1c2c3, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    
    # use the sobel edge detector on each channel of the c1c2c3 colour space, then merge the sobel values together in some way
    # shadows should have c1 c2 and c3 relatively unchanged. Also exclude pixels that have brightness > standard background.
    
    tiled.add(c1c2c3, "c1c2c3")
    additive_sob = sobel(c1c2c3[...,0])/3 + sobel(c1c2c3[...,1])/3 + sobel(c1c2c3[...,2])/3
    additive_sob = cv2.normalize(additive_sob, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    tiled.add(additive_sob, "c1c2c3_Sobel")

    thresh = cv2.threshold(additive_sob, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    tiled.add(thresh, "c1c2c3_Sobel Otsu_thresh")
    
    #would be good here to use a multi-channel edge detector, but Canny is not implemented for multiple channels. See http://stackoverflow.com/questions/8092059/color-edge-detection-opencv 

    
    im = cv2.bilateralFilter(img, 7, 250, 250).astype(np.float32)+0.001 #to avoid division by 0
    c1c2c3 = np.arctan(im/np.dstack((cv2.max(im[...,1], im[...,2]), cv2.max(im[...,0], im[...,2]), cv2.max(im[...,0], im[...,1]))))
    c1c2c3 = cv2.normalize(c1c2c3, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    # use the sobel edge detector on each channel of the c1c2c3 colour space, then merge the sobel values together in some way
    # shadows should have c1 c2 and c3 relatively unchanged. Also exclude pixels that have brightness > standard background.

    additive_sob = sobel(c1c2c3[...,0])/3 + sobel(c1c2c3[...,1])/3 + sobel(c1c2c3[...,2])/3
    additive_sob = cv2.normalize(additive_sob, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    tiled.add(additive_sob, "blur+c1c2c3_Sobel", newrow=True)

    thresh_val, thresh = cv2.threshold(cv2.bilateralFilter(additive_sob, 5, 40, 40), 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    tiled.add(thresh, "blur+c1c2c3_Sobel Otsu_thresh")

    thresh_val, thresh = cv2.threshold(additive_sob, thresh_val/2, 255, cv2.THRESH_BINARY)

    tiled.add(thresh, "blur+c1c2c3_Sobel Otsu_thresh/2")

    # use the sobel edge detector on each channel of the c1c2c3 colour space, then merge the sobel values together in some way
    # shadows should have c1 c2 and c3 relatively unchanged. Also exclude pixels that have brightness > standard background.

    # only subtract areas with +- same colour
    
    # but the background may be spotty, thus have sobel edges. To get around this, maybe use blur?
    pass

