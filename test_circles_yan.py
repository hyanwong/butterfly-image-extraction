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
#    c1img = cv2.threshold(c1img, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    c1img = cv2.adaptiveThreshold(c1img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, int(round((min(c1img.shape[0],c1img.shape[1])/16))*2+1) , 20)
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


def plot_circular_contour(c1img, accumulate_binary):
    c1img = cv2.adaptiveThreshold(c1img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, int(round((min(c1img.shape[0],c1img.shape[1])/16 ))*2+1), 20)
    smooth_contours = cv2.findContours(c1img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]


    for c1 in smooth_contours:
        c2 = cv2.approxPolyDP(c1,1,True)
        a1 = cv2.contourArea(c1)
        a2 = cv2.contourArea(c2)
        l1 = cv2.arcLength(c1, True)
        l2 = cv2.arcLength(c2, True)

        print(l1)
        if l1 > 25:
            Q1 = (4*np.pi*a1)/(l1*l1)
            if Q1 > 0.9:
                cv2.drawContours(accumulate_binary, [c1], -1, color=(255,255,255), thickness=cv2.cv.CV_FILLED)                

        if l2 > 25:
            Q2 = (4*np.pi*a2)/(l2*l2)
            if Q2>0.9:
                cv2.drawContours(accumulate_binary, [c2], -1, color=(255,255,255), thickness=cv2.cv.CV_FILLED)

    return accumulate_binary

img = cv2.imread("98261_580_360.jpg")
h, w = img.shape[0:2]
im = img.astype(np.float32)+0.001 #to avoid division by 0
c1c2c3 = np.arctan(im/np.dstack((cv2.max(im[...,1], im[...,2]), cv2.max(im[...,0], im[...,2]), cv2.max(im[...,0], im[...,1]))))
bimg,gimg,rimg = cv2.split(c1c2c3)
rimg =(cv2.normalize(rimg, rimg, 0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1))
gimg = (cv2.normalize(gimg, gimg, 0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1))
bimg = (cv2.normalize(bimg, bimg, 0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1))

savecontours(rimg, "r")
savecontours(gimg, "g")
savecontours(bimg, "b")

bin = np.zeros((img.shape[0], img.shape[1],3), np.uint8)
plot_circular_contour(rimg, bin)
plot_circular_contour(gimg, bin)
plot_circular_contour(bimg, bin)
cv2.imshow("Circles", bin)
cv2.waitKey(0)
