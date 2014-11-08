from __future__ import division
from __future__ import print_function
import cv2
import urllib
import numpy as np
import gspread
import random
import re
import os
import glob
import sys


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

class compound_img:
    '''make an image consisting of a number of pictures tiled together. By default add left to right. Use add(..., newrow=TRUE) to add a new row.'''
    main_image = None
    prev_right_edge = 0
    colours = [[255,0,0],[0,0,255], [0,255,0],[0,255,255],[255,0,255],[255,255,0]]

    def add(self, img, name=None, newrow=False, contours=None, focal_contours=[]):
        #first convert to BGRA with alpha channel, so we can overlay translucent contours
        if len(img.shape) < 3 or img.shape[2] == 1:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGRA)
        elif len(img.shape) < 4:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        else:
            img = img.copy()
            
        h, w = img.shape[:2]
    
        if self.main_image is None:
            self.main_image = np.zeros((h,w,3), np.uint8)

        if newrow:
            newrow = np.zeros((h,self.main_image.shape[1],3), np.uint8)
            self.main_image = np.concatenate((self.main_image,newrow)) #add enough space for another row
            self.prev_right_edge = 0

        width_diff = self.prev_right_edge + w - self.main_image.shape[1]
        if width_diff > 0:
            self.main_image = np.concatenate((self.main_image, np.zeros((self.main_image.shape[0],width_diff,3), np.uint8)), 1) #extend main.image width to allow for new pic
    
        if contours is not None:
            cv2.drawContours(img, contours, -1, self.colours[0] + [100])
            for i in range(len(focal_contours)):
                cv2.drawContours(img, contours, focal_contours[i], self.colours[i+1] + [200])
            
        if name is not None:
            cv2.putText(img, name, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)

        bottom = self.main_image.shape[0]
        self.main_image[(bottom-h):bottom, self.prev_right_edge:self.prev_right_edge+w] = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        self.prev_right_edge += w

def crop_frame(img, min_prop_picture_in_frame):
    '''Detect if picture is in a frame, by detecting rectangles and crop if any contain >75% of the image.
    This code does not seem to work consistently (e.g. on eol.org/data_objects/17762955): it probably needs tweaking'''
    h, w = img.shape[:2]
    epsilon = (h+w)/2 * 0.01 #allow lines in rectange detection to deviate about 1% of the size of the image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.Canny(grey_img, 0, 50, apertureSize=3)


    

    contours, hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_frame = -1
    best_area_frac = 1
    for j in range(len(contours)):
        hull = cv2.convexHull(contours[j], returnPoints=True)
        area_frac = cv2.contourArea(hull) / h / w
        if (area_frac > min_prop_picture_in_frame):
            approxCurve = cv2.approxPolyDP(hull, epsilon, True)
            
            frame=False
            
            if ((len(approxCurve) == 4) and cv2.isContourConvex(approxCurve)):
                #look to check that we have (approximately) [x1,y1], [x1,y2], [x2,y2], [x2,y1]
                #for some reason curve points are expressed as [n, 0, x] rather than [n,x]
                if (abs(approxCurve[0,0,0] - approxCurve[1,0,0]) < epsilon):
                    if((abs(approxCurve[1,0,1] - approxCurve[2,0,1]) < epsilon) and \
                       (abs(approxCurve[2,0,0] - approxCurve[3,0,0]) < epsilon) and \
                       (abs(approxCurve[3,0,1] - approxCurve[0,0,1]) < epsilon)):
                        frame = True
                elif (abs(approxCurve[0,0,1] - approxCurve[1,0,1]) < epsilon):
                    if((abs(approxCurve[1,0,0] - approxCurve[2,0,0]) < epsilon) and \
                       (abs(approxCurve[2,0,1] - approxCurve[3,0,1]) < epsilon) and \
                       (abs(approxCurve[3,0,0] - approxCurve[0,0,0]) < epsilon)):
                        frame = True
            if (frame):
                #find smallest rect that is still > 75%
                if (area_frac < best_area_frac):
                    best_frame = j
                    best_area_frac = area_frac
    if (best_frame >= 0):
        crop = cv2.boundingRect(contours[best_frame])
        crop_left = crop[0]
        crop_top = crop[1]
        crop_right = w-crop[2]
        crop_bottom = h-crop[3]
        return crop_left,crop_top,crop_right,crop_bottom
    else:
        return 0,0,0,0

def crop_border(img, edge_fraction=0.1, line_length = 0.8, maxLineGap=10, horiz_vert_max_gradient = 1/40):
    '''Detect if picture is in a frame, by flood filling from the top left corner, then looking for long horizontal or vertical lines in the resulting mask.
    Horizontal lines are defined as -horiz_vert_max_gradient < GRAD < horiz_vert_max_gradient. Verical lines -horiz_vert_max_gradient < 1/GRAD < horiz_vert_max_gradient
    We only choose lines within a certain fraction of the edge of the picture (10%, by default). If there are left, right, top AND bottom lines, assume it is a frame.
    Some test cases are eol.org/data_objects/17762955 '''

    h, w = img.shape[:2]
    flood_from = (0,0)     # flood from top left pixel colour
    flood_param =3.5
    mask = np.zeros((h+2,w+2,1), np.uint8)
    cv2.floodFill(img, mask, flood_from,0,(flood_param, flood_param, flood_param),(flood_param, flood_param, flood_param), cv2.FLOODFILL_MASK_ONLY)

    bw = mask[1:-1,1:-1,...]*255
    kern = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    edges = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kern) # find 
    
    best = [None,None,None,None]
    vert_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold =4, minLineLength=h* line_length, maxLineGap= maxLineGap)
    if vert_lines is not None:
        for l in range(vert_lines.shape[1]):
            p = vert_lines[0,l,:]
            if (p[1] != p[3]) and abs((p[0]-p[2])/(p[1]-p[3]))< horiz_vert_max_gradient: #vert has gradient 40 in 1 or steeper
                x_pos = np.mean(p[[0,2]])
                if   (best[0] is None or (x_pos > best[0])) and (x_pos < 0.1*w): #in leftmost 10% of image 
                    best[0] = np.min(p[[0,2]])
                    print("got x min")
                elif (best[2] is None or (x_pos < best[2])) and (x_pos > 0.9*w): #in rightmost 10% of image 
                    best[2] = np.max(p[[0,2]])
                    print("got x max")
    horiz_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold =4, minLineLength=w* line_length, maxLineGap= maxLineGap)
    if horiz_lines is not None:
        for l in range(horiz_lines.shape[1]):
            p = horiz_lines[0,l,:]
            if (p[0] != p[2]) and abs((p[1]-p[3])/(p[0]-p[2]))< horiz_vert_max_gradient: #vert has gradient 1 in 40 or shallower
                y_pos = np.mean(p[[1,3]])
                if   (best[1] is None or (y_pos > best[1])) and (y_pos < 0.1*h): #in topmost 10% of image 
                    best[1] = np.min(p[[1,3]])
                    print("got y min")
                elif (best[3] is None or (y_pos < best[3])) and (y_pos > 0.9*h): #in bottommost 10% of image 
                    best[3] = np.max(p[[1,3]])
                    print("got y max")


    if all(i is not None for i in best):
        print("Cropped")
        return best[0], best[1], w-best[2], h-best[3]
    else:
        return 0,0,0,0


def save_Hu_moments(thresholded, EoLobjectID, contour_dir, filename):
    '''Save to file the 7 Hu moments for each contour in each image for statistical analysis (e.g. for analysis to predict
    which are butterfly shaped). Also save images of each contour so we can look through and mark by hand which are the
    butterfly outlines'''
    if not hasattr(save_Hu_moments, "writefile"):
        save_Hu_moments.writefile = open(os.path.join(contour_dir,filename), 'w')  # it doesn't exist yet, so initialize it
        save_Hu_moments.writefile.write("img-contour	crude.points	simp.crude.points	smooth.points	simp.smooth.points	area	hu1	hu2	hu3	hu4	hu5	hu6	hu7\n")

    crude_contours = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    smooth_contours = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    if len(crude_contours) != len(smooth_contours):
        print("oops - the two contour methods should give the same number of contour areas")
        exit

    for i in range(len(smooth_contours)):
        if cv2.contourArea(smooth_contours[i]) > 0.001*thresholded.shape[0]*thresholded.shape[1]: #only pick areas > 0.1% of the area
            roi = np.asarray(cv2.boundingRect(smooth_contours[i]))
            img = np.zeros((roi[3], roi[2],3), np.uint8)
            cv2.drawContours(img, [smooth_contours[i]-roi[0:2]], 0, (255,255,255), 2)
            cv2.imwrite(os.path.join(contour_dir, "{}-{}.jpg".format(EoLobjectID, i)), img)
#            np.save(os.path.join(contour_dir, "{}-{}.npy".format(EoLobjectID, i)), smooth_contours[i]) #save the contour coordinates
            
            Hu_text = "\t".join(np.char.mod('%e', cv2.HuMoments(cv2.moments(smooth_contours[i]))).flatten())
            contour_lengths = "\t{}\t{}\t{}\t{}".format(len(crude_contours[i]), len(cv2.approxPolyDP(crude_contours[i],1,True)), len(smooth_contours[i]), len(cv2.approxPolyDP(smooth_contours[i],1,True)))
            save_Hu_moments.writefile.write("{}-{}\t{}\t{}\t{}\n".format(EoLobjectID ,i, contour_lengths, cv2.contourArea(smooth_contours[i]), Hu_text))
            save_Hu_moments.writefile.flush()
        
def prob_butterfly(crude_contours, smooth_contours, eqn=0):
    '''This is based on logistic regression of Hu moments for known butterfly and non-butterfly shapes. The analysis can be done by calling 
    save_Hu_moments(mask_after_grabcut, EoLobjectID) to save the Hu moments for contours from a number of images to a file. Once the correct butterfly contours have
    been identified by hand, and their names (as ObjID-ContourNum) stored in a file, say "butterflies.data", the analysis can then be done in R by calling
        Hu.data <- read.delim("Hu.data", row.names=1)
        Hu.data$butterfly <- 0
        Hu.data[scan("butterflies.data", "character"), "butterfly"] <- 1
        Hu.data$logh6 <- ifelse(Hu.data$hu6 < 1e-11, log(1e-11), log(Hu.data$hu6)); #6th Hu moment is useful when logged, but needs truncation to avoid neg numbers. Try hist(Hu.data$logh6) to inspect
        model <- glm(butterfly ~ hu1 + I(hu1^2) + logh6+ I(log(smooth.points)/log(crude.points)), Hu.data, family="binomial")
        summary(model)
        complex_model <- glm(butterfly ~ poly(hu1, 2) + logh6+ (log(smooth.points)+log(crude.points)+log(simp.crude.points))^3, Hu.data, family="binomial")
        summary(complex_model)
        simple_model <- glm(butterfly ~ log(hu1) + I(log(hu1)^2) + log(hu2) + I(log(hu2)^2) + hu4 , Hu.data, family="binomial")
        summary(simple_model)
    This should give something like

                                          Estimate Std. Error z value Pr(>|z|)    
(Intercept)                              -72.49121   11.65517  -6.220 4.98e-10 ***
hu1                                      338.98944   89.40650   3.792 0.000150 ***
I(hu1^2)                                -698.18014  194.39183  -3.592 0.000329 ***
logh6                                     -0.27832    0.05244  -5.307 1.11e-07 ***
I(log(smooth.points)/log(crude.points))   35.06491    5.94520   5.898 3.68e-09 ***

    Null deviance: 551.20  on 451  degrees of freedom
Residual deviance: 107.77  on 447  degrees of freedom
AIC: 117.77


or for the more complex model

or omitting the jaggedness param

(Intercept)   -87.185     13.941  -6.254 4.00e-10 ***
hu1           710.382    132.003   5.382 7.38e-08 ***
I(hu1^2)    -1369.683    313.453  -4.370 1.24e-05 ***
hu2            -9.629     77.719  -0.124 0.901402    
I(hu2^2)    -7695.691   2987.023  -2.576 0.009984 ** 
hu4         -4701.824   1352.317  -3.477 0.000507 ***
---
    Null deviance: 476.24  on 382  degrees of freedom
Residual deviance: 161.16  on 377  degrees of freedom
AIC: 173.16


    '''
    if len(crude_contours) != len(smooth_contours):
        print("oops - the two contour methods should give the same number of contour areas")
        exit

    values = np.zeros(len(smooth_contours))
    for i in range(len(smooth_contours)):
        if len(crude_contours[i])==1: #to avoid div by 0 in smoothness calc. This is a pointless contour anyway
            values[i]=0
        else:
            Hu = cv2.HuMoments(cv2.moments(smooth_contours[i]))
            hu1 = Hu[0]
            hu2 = Hu[1]
            hu4 = Hu[3]
            if(Hu[5] < 1e-11):
                log_hu6 = np.log(1e-11)
            else:
                log_hu6 = np.log(Hu[5])
            smoothness = np.log(len(smooth_contours[i]))/np.log(len(crude_contours[i])) #should help weed out contours with lots of straight lines

            if eqn==0:
                x = -82.70066 +417.23428*hu1 -862.08239*hu1**2 -0.31457*log_hu6 + 35.49917*smoothness
            elif eqn==1:
                epsilon=1e-4
                x = -334.561 -142.494*np.log(hu1+epsilon) -48.361*np.log(hu1+epsilon)**2 -46.982*np.log(hu4+epsilon) -2.762*np.log(hu4+epsilon)**2 + 41.824*smoothness
            elif eqn==3:
                x = -86.185 + 710.382*hu1   -1369.683*hu1**2 -9.629*hu2 -7695.691*hu2**2 -4701.824*hu4
            values[i] = np.exp(x)/(1+np.exp(x))
        
    return values


#def find_background_using_meanshift(img, ):
    # 

def find_background_using_floodfill(img, quantized_img, n_flood_areas_for_cutoff, n_flood_areas_max, flood_parameters = [8,5], flood_type = cv2.FLOODFILL_FIXED_RANGE, reflood=True):
    '''Use flood filling to detect if there is a background of a constant shade. First try flooding from num_points around the (massively despeckled) image, and pick the largest flooded area,
    assuming this is the background colour (use fixed range flooding to make sure we do not e.g. grade into unfocussed brown branches against a green background).
    Fill that area with the average colour within it, and reflood, to grab any extraneous pixels. Repeat, adding a new flood area to the background only if its 
    average colour is within a certain tolerance of the background area. After n_flood_areas_for_cutoff repeats, save the total area covered, to use 
    as a cutoff to decide if this is a picture with a uniform background. 
    
    Now that we can decide if background is uniform, catch any extra parts by flood more areas using a floating range, up to n_flood_areas_max times.use the total background as a mask
    '''
    # PARAMETERS TO TWEAK
    num_points=50
    max_flood_area_value_diff = 50 #only add another area if within this value of the original
    max_flood_area_brightness_diff = 0.4 #only add another area if within this brightness of the original
    max_flood_area_chromocity_diff = 1.5 #only add another area if within this chromacity of the original

    h, w = img.shape[:2]
    full_mask = np.zeros((h+2,w+2,1), np.uint8)
    blank = np.zeros((h,w,3), np.uint8)
    best_f = []    

    for j in range(n_flood_areas_max): #pick the n largest areas (that have the same approximate colour)
        full_mask[full_mask!=0] += 1  #by default, masks contains '1' where flooded. At start of loop, increment all non-zero numbers by 1
        bestflood = 0
        for k in range(num_points):
            x = int(abs(k/num_points*w*2.01-(w-1)))
            y = int((k/num_points*h*4.01-2*(h-1)) % h) # tile the points in a pattern that looks like XX
#            cv2.circle(test_img, (x,y), 4, (255,0,0), 2);

            # look at the colour of the focal pixel: by trial and error it seems that pinned butterfly backgrounds
            # can contain substantial amounts of blue. If we ignore absolute blue levels, and plot
            # X=R/mean(B,G,R) vs Y=G/mean(R,G,B) then pinned backgrounds mostly lie in the lower left quadrant
            # at x<1.1 && y<1.1. Outside of this region we can be more stringent with the flood parameter
            focal_pix = 1e-4 + img[y,x,:] #add a tiny value to avoid division by zero errors
            X = (focal_pix[2])/np.mean(focal_pix)            
            Y = (focal_pix[1])/np.mean(focal_pix)         
            if ((X<1.1) and (Y<1.1)):
                f=flood_parameters[0]
            else:
                f=flood_parameters[1]

            # get the first set of pixels & mask
            mask = full_mask.copy() #copy bits in to mask
            cv2.floodFill(quantized_img, mask,(x,y),0,(f,f,f),(f,f,f), \
                cv2.FLOODFILL_MASK_ONLY | flood_type) 

            if reflood:
                # now actually fill that area with the av colour
                mask[mask != 1] = 0
                av_colour = cv2.mean(quantized_img, mask = mask[1:-1,1:-1])
                av_colour_uint8 = np.array(av_colour[0:3], dtype=np.uint8)
            
                temp = quantized_img.copy()
                temp = cv2.bitwise_or(blank, av_colour_uint8, dst = temp, mask = mask[1:-1,1:-1]) #only overwrite where mask == 1
            
                # try filling again from the same point
                mask[:] = full_mask
                cv2.floodFill(temp, mask,(x,y),0,(f,f,f),(f,f,f), \
                    cv2.FLOODFILL_MASK_ONLY | flood_type)
            
            count = cv2.countNonZero(mask[1:-1,1:-1]) / h / w
            if (count > bestflood):
                bestflood = count
                best_f.append(f) #only needed for outputting the initial flood parameter
                temp_mask = mask #store the current best mask
        full_mask = temp_mask

        if j==0:
            print("regions added to mask: 0 ({:1.2f}%)".format(bestflood*100), end="");
            good_mask = full_mask.copy()

        else: #not on first iteration: should we add this new area to the best mask?
            #get the av colour of the area filled on all previous iterations
            old_mean, old_stdev = cv2.meanStdDev(img, mask = good_mask[1:-1,1:-1])
            old_mean = old_mean.ravel() #bizarrely, meanStdDev returns an array of size [3,1], not [3]
            old_stdev = old_stdev.ravel()

            #get the av colour of the area filled this time (should be bit == 1)
            new_mean, new_stdev = cv2. meanStdDev(img, mask=np.where((full_mask==1),1,0).astype(np.uint8)[1:-1,1:-1])
            new_mean = new_mean.ravel() #bizarrely, meanStdDev returns an array of size [3,1], not [3]
            new_stdev = new_stdev.ravel()
            
            #brightness_distortion = 1 means no brightness difference between known background and new.
            #by trial and error, log(brightness_distortion) +- 0.3 covers most background variation (not flash shadows)
            
#            if (abs(value_diff(new_mean, old_mean)) <  max_flood_area_value_diff):
            if (abs(np.log(brightness_distortion(new_mean, old_mean, old_stdev)+1e-10)) <  max_flood_area_brightness_diff):
                if (chromacity_distortion(new_mean, old_mean, old_stdev) <  max_flood_area_chromocity_diff) :
                    good_mask[full_mask==1] = 1
                    if j<n_flood_areas_for_cutoff:
                        print(" {} ({:1.2f}%)".format(j, bestflood*100), end='');
                    else:
                        print(" [{} ({:1.2f})]".format(j, bestflood*100), end='');


               
        if j==(n_flood_areas_for_cutoff-1):
            cutoff_mask = good_mask.copy()[1:-1,1:-1]

    good_mask = good_mask[1:-1,1:-1] #trim off the extra pixels at the edge of the mask to make it the same as the image
    print(". Total area flooded = {:0.2f} %.".format(cv2.countNonZero(good_mask) / h / w *100))

    return good_mask, cutoff_mask, best_f[0]
    
    
def refine_background_via_grabcut(img, is_background, dilate=False):
    #use grabcut (http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html) 
    # to cut out other background pixels
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,img.shape[1],img.shape[0])
    grabcut_mask = np.where(is_background!=0,cv2.GC_BGD,cv2.GC_PR_FGD).astype(np.uint8) #background should be 0, probable foreground = 3 
    cv2.grabCut(img, grabcut_mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    return np.where((grabcut_mask ==2)|(grabcut_mask ==0),0,1).astype(np.uint8)


def find_butterfly(thresholded):
    '''Find all contours in the thresholded image, and for each contour, use the Hu moments, plus an estimate of the proportion of the contour that consists of straight lines,
    as predictors of the probability that a contour represents a butterfly shape. For details of the model, see the function prob_butterfly()'''
    cutoff = 0.001*thresholded.shape[0]*thresholded.shape[1]
    thresholded = cv2.normalize(thresholded, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    crude_contours = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    smooth_contours= cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    prob = []
    index = []
        
    for i in [0]: #different numbers represent different statistical models used in prob_butterfly
        pr = prob_butterfly(crude_contours, smooth_contours, i)
        p_large = [pr[x] if cv2.contourArea(smooth_contours[x]) > cutoff else 0 for x in range(len(pr))]
        prob.append(np.max(p_large))
        index.append(np.argmax(p_large))
    return prob, index, crude_contours


#####BEGIN  EDITS HERE TO FIND WHERE CORNELL (GRAY) IMAGES FAIL

def grab_circle(small_img, large_img, EoLobjectID, param_dir = None, composite_file_dir = True, butterfly_with_contour_file_dir = "butterflies"):
    '''Process a small and a large butterfly file for circle detection'''
    H, W = large_img.shape[:2]
    h, w = small_img.shape[:2]
    myradii = []
    myxy = []

    # First crop any exterior frames (only if inner rect > 60% of picture)
    crop_left,crop_top,crop_right,crop_bottom = crop_frame(small_img, 0.6)
    img = small_img[crop_top:(h-crop_bottom), crop_left:(w-crop_right),:]
    h1, w1 = img.shape[:2]
    emptymask = np.zeros((h1,w1,1), np.uint8)
    contourmask = np.zeros((h1,w1,1), np.uint8)
    contourmaskr = np.zeros((h1,w1,1), np.uint8)
    contourmaskg = np.zeros((h1,w1,1), np.uint8)
    contourmaskb = np.zeros((h1,w1,1), np.uint8)

    #output image - compound
    full = compound_img()
    full.add(img, "Original")

    #look at individual channels
    im = img.astype(np.float32)+0.001 #to avoid division by 0
    c1c2c3 = np.arctan(im/np.dstack((cv2.max(im[...,1], im[...,2]), cv2.max(im[...,0], im[...,2]), cv2.max(im[...,0], im[...,1]))))
    bimg,gimg,rimg = cv2.split(c1c2c3)
    bimg = cv2.normalize(bimg,0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1)
    gimg = cv2.normalize(gimg,0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1)
    rimg = cv2.normalize(rimg,0,255,cv2.NORM_MINMAX,dtype=cv2.cv.CV_8UC1)
    #bimg = bimg.astype(np.int)
    #gimg = gimg.astype(np.int)
    #rimg = rimg.astype(np.int)


    output1 = cv2.merge((bimg,bimg,bimg))
    output2 = cv2.merge((gimg,gimg,gimg))
    output3 = cv2.merge((rimg,rimg,rimg))
    myxs = []
    myys = []
    myradii = []
    mycols = []

    #process blue channel
    #threshb, img_bw = cv2.threshold(bimg, 5, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshb, img_bw = cv2.threshold(bimg,245,255,cv2.THRESH_TRUNC)
    despeckled = cv2.bilateralFilter(img_bw, 5, 100, 100)
    img_bw = cv2.bilateralFilter(despeckled, 7, 50, 50)
    #outputb = cv2.merge((bimg,bimg,bimg))
    outputb2 = cv2.merge((img_bw,img_bw,img_bw))
    temp_contours,hier= cv2.findContours(img_bw, cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    contour_test = range(len(temp_contours))
    closed_contours = []
    mycontours = []
    for i in range(len(temp_contours)):
        temp = temp_contours[i]
        (x,y),radius = cv2.minEnclosingCircle(temp)
        if radius > h/50:
            mycontours.append(temp)
            contour_test[i] = False
            temp_poly = cv2.approxPolyDP(temp,3,closed=1)
            ctrarea = cv2.contourArea(temp) #USING TEMP_poly IS WORSE
            ctrlength = cv2.arcLength(temp, closed=1) #USING TEMP_poly IS WORSE
            ctrcompact = (4 * np.pi * ctrarea) / (pow(ctrlength, 2))
            if ctrcompact > 0.85:
                closed_contours.append(temp_contours[i])                 
            elif temp_poly.size > 6 and ctrarea>0:
                if abs( (ctrarea/np.pi) - np.power(ctrlength/(2*np.pi),2))/(ctrarea/np.pi) <= 0.25:
                    contour_test[i] = True
                    closed_contours.append(temp_contours[i])
    for i in range(len(closed_contours)):
        temp = closed_contours[i]
        (x,y),radius = cv2.minEnclosingCircle(temp)
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #M = cv2.moments(temp)
        myxs.append(x)
        myys.append(y)
        myradii.append(radius)
        mycols.append("blue")
        cv2.circle(output1, (int(x),int(y)),int(radius/2),(255,0,0),4)

    circles1 = cv2.HoughCircles(bimg, cv2.cv.CV_HOUGH_GRADIENT, 12,np.int(h/10),2,50,50,np.int(h/300),np.int(h/20));
    if circles1 is not None:
        circles1 = np.round(circles1[0,:]).astype("int")
        #print circles_b.shape
        #Draw the circles detected
        for (x, y, r) in circles1:
            if r>h/40:
                cv2.circle(output2, (int(x),int(y)),int(r/2),(255,0,0),4)
                myxs.append(x)
                myys.append(y)
                myradii.append(r)
                mycols.append("blue")

        #full.add(output1, "Contour Channel 2")

    #process green channel
    #threshb, img_bw = cv2.threshold(gimg, 15, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshg, img_bw = cv2.threshold(gimg,245,255,cv2.THRESH_TRUNC)
    despeckled = cv2.bilateralFilter(img_bw, 5, 100, 100)
    img_bw = cv2.bilateralFilter(despeckled, 7, 50, 50)
    outputg = cv2.merge((gimg,gimg,gimg))
    outputg2 = cv2.merge((img_bw,img_bw,img_bw))
    temp_contours,hier= cv2.findContours(img_bw, cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    contour_test = range(len(temp_contours))
    closed_contours = []
    mycontours = []
    for i in range(len(temp_contours)):
        temp = temp_contours[i]
        (x,y),radius = cv2.minEnclosingCircle(temp)
        if radius > h/50:
            mycontours.append(temp)
            contour_test[i] = False
            temp_poly = cv2.approxPolyDP(temp,3,closed=1)
            ctrarea = cv2.contourArea(temp) #USING TEMP_poly IS WORSE
            ctrlength = cv2.arcLength(temp, closed=1) #USING TEMP_poly IS WORSE
            ctrcompact = (4 * np.pi * ctrarea) / (pow(ctrlength, 2))
            if ctrcompact > 0.85:
                closed_contours.append(temp_contours[i])                 
            elif temp_poly.size > 6 and ctrarea>0:
                if abs( (ctrarea/np.pi) - np.power(ctrlength/(2*np.pi),2))/(ctrarea/np.pi) <= 0.25:
                    contour_test[i] = True
                    closed_contours.append(temp_contours[i])
    for i in range(len(closed_contours)):
        temp = closed_contours[i]
        (x,y),radius = cv2.minEnclosingCircle(temp)
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #M = cv2.moments(temp)
        myxs.append(x)
        myys.append(y)
        myradii.append(radius)
        mycols.append("green")
        cv2.circle(output1, (int(x),int(y)),int(radius/2),(0,255,0),3)


    circles1 = cv2.HoughCircles(gimg, cv2.cv.CV_HOUGH_GRADIENT, 12,np.int(h/10),2,50,50,np.int(h/300),np.int(h/20));
    if circles1 is not None:
        circles1 = np.round(circles1[0,:]).astype("int")
        #print circles_b.shape
        #Draw the circles detected
        for (x, y, r) in circles1:
            if r>h/40:
                cv2.circle(output2, (int(x),int(y)),int(r/2),(0,255,0),3)
                myxs.append(x)
                myys.append(y)
                myradii.append(r)
                mycols.append("green")
        #full.add(output1, "Contour Channel 2")
    


    #process red channel
    threshr, img_bw = cv2.threshold(rimg,245,255,cv2.THRESH_TRUNC)
    despeckled = cv2.bilateralFilter(img_bw, 5, 100, 100)
    img_bw = cv2.bilateralFilter(despeckled, 7, 50, 50)
    outputr = cv2.merge((rimg,rimg,rimg))
    outputr2 = cv2.merge((img_bw,img_bw,img_bw))
    temp_contours,hier= cv2.findContours(img_bw, cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    contour_test = range(len(temp_contours))
    closed_contours = []
    mycontours = []
    for i in range(len(temp_contours)):
        temp = temp_contours[i]
        (x,y),radius = cv2.minEnclosingCircle(temp)
        if radius > h/50:
            mycontours.append(temp)
            contour_test[i] = False
            temp_poly = cv2.approxPolyDP(temp,3,closed=1)
            ctrarea = cv2.contourArea(temp) #USING TEMP_poly IS WORSE
            ctrlength = cv2.arcLength(temp, closed=1) #USING TEMP_poly IS WORSE
            ctrcompact = (4 * np.pi * ctrarea) / (pow(ctrlength, 2))
            if ctrcompact > 0.85:
                closed_contours.append(temp_contours[i])                 
            elif temp_poly.size > 6 and ctrarea>0:
                if abs( (ctrarea/np.pi) - np.power(ctrlength/(2*np.pi),2))/(ctrarea/np.pi) <= 0.25:
                    contour_test[i] = True
                    closed_contours.append(temp_contours[i])
    for i in range(len(closed_contours)):
        temp = closed_contours[i]
        (x,y),radius = cv2.minEnclosingCircle(temp)
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #M = cv2.moments(temp)
        myxs.append(x)
        myys.append(y)
        myradii.append(radius)
        mycols.append("red")
        cv2.circle(output1, (int(x),int(y)),int(radius/2),(0,0,255),2)

    circles1 = cv2.HoughCircles(rimg, cv2.cv.CV_HOUGH_GRADIENT, 12,np.int(h/10),2,50,50,np.int(h/300),np.int(h/20));
    if circles1 is not None:
        circles1 = np.round(circles1[0,:]).astype("int")
        #print circles_b.shape
        #Draw the circles detected
        for (x, y, r) in circles1:
            if r>h/40:
                cv2.circle(output2, (int(x),int(y)),int(r/2),(0,0,255),2)
                myxs.append(x)
                myys.append(y)
                myradii.append(r)
                mycols.append("red")
    
    full.add(output1, "From Contours")
    full.add(output2, "From HoughCircles")
    full.add(output3, "Red grayscale")
    full.add(outputb2, "Blue Binary", newrow=True)
    full.add(outputg2, "Green Binary")
    full.add(outputr2, "Red Binary")


    cv2.imshow("output",full.main_image)
    cv2.waitKey(0);


    #cv2.imshow( "output", np.hstack([img,outputb,outputg,outputr] ))

################## main script here

contour_dir = None #"contours" #set to None unless you want to output params to model probabbility that a contour outline is a butterfly.
folders = ["classification", "butterflies"];
for folder in folders:
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print(e)

image_folder = 'images_eol' #this should contain ID_580_360.jpg files together with the full-sized ID.xxx files. If it is empty or does not exist, get them from google docs
if os.path.isdir(image_folder):
    pattern = re.compile("(.*)_580_360.jpg$"); #_580_360
    for img_file in os.listdir(image_folder):
        match = pattern.search(img_file)
        if match is not None:
            tmpID = match.group(1)
            small_file = os.path.join(image_folder, img_file)
            match_glob = os.path.join(image_folder, tmpID+".*");
            large_files = glob.glob(match_glob)
            if len(large_files) ==1:
                print("opening {} and {} (ID={})".format(small_file, large_files[0], tmpID))
                grab_circle(cv2.imread(small_file, cv2.CV_LOAD_IMAGE_COLOR), cv2.imread(large_files[0], cv2.CV_LOAD_IMAGE_COLOR), tmpID, contour_dir, folders[0], folders[1])
            else:
                print("problem with opening {}: found {} files when matching {}".format(small_file, len(large_files), match_glob))                
else:
    gc = gspread.login("EOLBHL2014","EoL/BHL2014")
    sh = gc.open_by_key("0AsbkF6jVHju6dGttX1NoWmpoM0d3RDgyN2ZROHp6enc") #this is the 35000 row spreadsheet
    #sh = gc.open_by_key("0AsbkF6jVHju6dGVKYUpiQmpDbjRweVo3YUNkeG9adEE") #this is the test spreadsheet
    worksheet = sh.get_worksheet(0)
    image_IDs = worksheet.col_values(1)
    URLs_1 = worksheet.col_values(2)
    URLs_2 = worksheet.col_values(3)

    random.seed(123);
    test_rows = random.sample(range(len(image_IDs)-1), 400)
    print("using rows ", end="")
    print(", ".join([str(x) for x in test_rows]))
    for row in test_rows:
        i=row+1; #miss the first (header) row
        print("Data_object {}: opening {}".format(image_IDs[i], URLs_1[i])) #to download these, try perl -ne 'if (/^Data_object (\d+): opening ([^_]*(.*)?\.(\w+))$/) {system "wget -O $1$3.$4 $2"}'
        req1 = urllib.urlopen(URLs_1[i])
        arr1 = np.asarray(bytearray(req1.read()), dtype=np.uint8)
        print("Data_object {}: opening {}".format(image_IDs[i], URLs_2[i]))
        req2 = urllib.urlopen(URLs_2[i])
        arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)
        #grab_butterfly(cv2.imdecode(arr1,cv2.CV_LOAD_IMAGE_COLOR), cv2.imdecode(arr2,cv2.CV_LOAD_IMAGE_COLOR), image_IDs[i], contour_dir, folders[0], folders[1])

