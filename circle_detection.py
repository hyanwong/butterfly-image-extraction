from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os

def savecontours(c1img, name):
    folder = "circlecontours/{}".format(name)
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        os.unlink(file_path)
        
    cv2.imshow(name, c1img)
    #c1img = cv2.threshold(c1img, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #parameter = 1/8 of minimum dimension of image - must be an odd number
    window_width = int(round((min(c1img.shape[0],c1img.shape[1])/16 ))*2+1)
    #cv2.ADAPTIVE_THRESH_MEAN_C worked better than cv2.ADAPTIVE_THRESH_GAUSSIAN_C with these parameters
    c1img = cv2.adaptiveThreshold(c1img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_width, 20)
    cv2.imshow("{} thresh".format(name), c1img)
    cv2.waitKey(0)

    smooth_contours = cv2.findContours(c1img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]

    img = np.zeros((c1img.shape[0], c1img.shape[1],3), np.uint8)
    cv2.drawContours(img, smooth_contours, -1, (255,255,255), 1)
    cv2.imwrite(os.path.join(folder, "all.png"), img)

    i=0
    for c1 in smooth_contours:
        c2 = cv2.approxPolyDP(c1,1,True)
        a1 = cv2.contourArea(c1)
        a2 = cv2.contourArea(c2)
        l1 = cv2.arcLength(c1, True)
        l2 = cv2.arcLength(c2, True)


        if l1 > 25:
            Q1 = (4*np.pi*a1)/(l1*l1)
            img = np.zeros((c1img.shape[0], c1img.shape[1],3), np.uint8)
            cv2.drawContours(img, [c1], -1, color=(255,255,255), thickness=1)
            cv2.imwrite(os.path.join(folder, "plain{:.4f}_contour{}.png".format(Q1, i)), img)

        if l2 > 25:
            Q2 = (4*np.pi*a2)/(l2*l2)
            img = np.zeros((c1img.shape[0], c1img.shape[1],3), np.uint8)
            cv2.drawContours(img, [c2], -1, color=(255,255,255), thickness=1)
            cv2.imwrite(os.path.join(folder, "approx{:.4f}_contour{}.png".format(Q2, i)), img)
        i = i+1


def plot_circular_contour(c1img, accumulate_binary, display=False):
    if display:
        display_image=np.zeros((c1img.shape[0], c1img.shape[1]*3), np.uint8)
        display_image[:,0:c1img.shape[1]] = c1img

    accum_params = []
    #parameter = 1/8 of minimum dimension of image - must be an odd number
    window_width = int(round((min(c1img.shape[0],c1img.shape[1])/16 ))*2+1)
    #cv2.ADAPTIVE_THRESH_MEAN_C worked better than cv2.ADAPTIVE_THRESH_GAUSSIAN_C with these parameters
    c1img = cv2.adaptiveThreshold(c1img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_width, 20)
    smooth_contours = cv2.findContours(c1img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    for c1 in smooth_contours:
        c2 = cv2.approxPolyDP(c1,1,True)
        a1 = cv2.contourArea(c1)
        a2 = cv2.contourArea(c2)
        l1 = cv2.arcLength(c1, True)
        l2 = cv2.arcLength(c2, True)

        #print(l1)
        #if l1 > 40: #removed; does not give any unique circles 
        if l2 > 45: #adjusted up from 25
            Q2 = (4*np.pi*a2)/(l2*l2)
            if Q2 > 0.8: #adjusted down from 0.9
                cv2.drawContours(accumulate_binary, [c2], -1, color=(255,255,255), thickness=cv2.cv.CV_FILLED)
                if display:
                    cv2.drawContours(display_image[:,(c1img.shape[1]*2):(c1img.shape[1]*3)], [c2], -1, color=(255,255,255), thickness=cv2.cv.CV_FILLED)

    if display:
        display_image[:,c1img.shape[1]:(c1img.shape[1]*2)] = c1img
        cv2.imshow("c1c1c1, thresh, mask", display_image) #ADDED COMMA
        cv2.waitKey()


##main - accesses images in folder

def best_circles(image_580_360, display=False):
    '''returns a binary mask image'''
    h, w = image_580_360.shape[0:2]
    img = image_580_360 #ADDED
    im = img.astype(np.float32)+0.001 #to avoid division by 0
    c1c2c3 = np.arctan(im/np.dstack((cv2.max(im[...,1], im[...,2]), cv2.max(im[...,0], im[...,2]), cv2.max(im[...,0], im[...,1]))))
    bimg,gimg,rimg = cv2.split(c1c2c3)
    rimg =(cv2.normalize(rimg, rimg, 0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1))
    gimg = (cv2.normalize(gimg, gimg, 0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1))
    bimg = (cv2.normalize(bimg, bimg, 0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1))
    accumulation_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    plot_circular_contour(rimg, accumulation_mask, display)
    plot_circular_contour(gimg, accumulation_mask, display)
    plot_circular_contour(bimg, accumulation_mask, display)
    
    #find circles in the accumulated mask
    contours = cv2.findContours(accumulation_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    if display:
        display_image=cv2.merge((accumulation_mask.copy(),accumulation_mask.copy(),accumulation_mask.copy())) ## Create a 3-channel display image

    for c in contours:
        (cx,cy),cr = cv2.minEnclosingCircle(c)
        print("{} {} {}".format(cx,cy,cr)) #ADDED PARENTHESES
        if display:
            cv2.circle(display_image, (int(cx),int(cy)),int(cr*.75), color=(0,0,255), thickness=cv2.cv.CV_FILLED) ## Add red circles

    if display:
        cv2.imshow("found circles", display_image) #ADDED COMMA
        cv2.waitKey()
    
    return(accumulation_mask)
