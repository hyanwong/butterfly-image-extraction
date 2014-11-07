from __future__ import division
import cv2
import numpy as np


def value_diff(I, mu):
    '''we pass in a numpy array: with the R, G, B components being the innermost'''
    if len(mu.shape) == 1:
        mu = np.rint(mu[np.newaxis,np.newaxis,:]).astype(np.uint8)
    if len(I.shape) == 1:
        I = np.rint(I[np.newaxis,np.newaxis,:]).astype(np.uint8)
    return(np.squeeze(cv2.subtract(cv2.cvtColor(I, cv2.COLOR_BGR2HSV)[...,2], cv2.cvtColor(mu, cv2.COLOR_BGR2HSV)[...,2], dtype=cv2.CV_32S)))

def brightness_distortion(I, mu, sigma):
    ''' From eqn 5 of Horprasert et. al. (1999) (http://vast.uccs.edu/~tboult/frame/Horprasert/HorprasertFRAME99.pdf)
    we pass in a numpy array: with the R, G, B components being the innermost'''
    return np.nansum(I*mu/sigma**2, axis=-1) / np.nansum((mu/sigma)**2, axis=-1)

def chromacity_distortion(I, mu, sigma):
    ''' From eqn 6 of Horprasert et. al. (1999) (http://vast.uccs.edu/~tboult/frame/Horprasert/HorprasertFRAME99.pdf)
    Note that the "+" signs are missing from this equation: the correct version is rewritten in http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3279231/'''
    alpha = brightness_distortion(I, mu, sigma)[...,None]
    return np.sqrt(np.nansum(((I - alpha * mu)/sigma)**2, axis=-1))

def sobel(single_channel_img, kernel_size = 1):
    '''standard Sobel edge detection algorithm: works well on c1c2c3 colour channels'''
    sob_x = cv2.Sobel(single_channel_img, cv2.CV_16S, 1, 0, ksize= kernel_size)
    sob_y = cv2.Sobel(single_channel_img, cv2.CV_16S, 0, 1, ksize= kernel_size)
    return cv2.addWeighted(cv2.convertScaleAbs(sob_x), 0.5, cv2.convertScaleAbs(sob_y), 0.5, 0)

def sobel_thresh(single_channel_img, kernel_size = 1, thresh=10):
    #return cv2.threshold(sobel(single_channel_img), thresh, 1, cv2.THRESH_BINARY)[1]
    return cv2.adaptiveThreshold(sobel(single_channel_img, kernel_size), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -3)
