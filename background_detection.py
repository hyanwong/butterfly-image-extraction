from __future__ import division
from __future__ import print_function
import cv2
import numpy as np

import basic_imgproc

#def using_meanshift(img, ):
    # 

def using_floodfill(img, quantized_img, n_flood_areas_for_cutoff, n_flood_areas_max, flood_parameters = [8,5], flood_type = cv2.FLOODFILL_FIXED_RANGE, reflood=True, verbose=False):
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
            if verbose:
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
            if (abs(np.log(basic_imgproc.brightness_distortion(new_mean, old_mean, old_stdev)+1e-10)) <  max_flood_area_brightness_diff):
                if (basic_imgproc.chromacity_distortion(new_mean, old_mean, old_stdev) <  max_flood_area_chromocity_diff) :
                    good_mask[full_mask==1] = 1
                    if verbose:
                        if j<n_flood_areas_for_cutoff:
                            print(" {} ({:1.2f}%)".format(j, bestflood*100), end='');
                        else:
                            print(" [{} ({:1.2f})]".format(j, bestflood*100), end='');


               
        if j==(n_flood_areas_for_cutoff-1):
            cutoff_mask = good_mask.copy()[1:-1,1:-1]

    good_mask = good_mask[1:-1,1:-1] #trim off the extra pixels at the edge of the mask to make it the same as the image
    if verbose:
        print(". Total area flooded = {:0.2f} %.".format(cv2.countNonZero(good_mask) / h / w *100))

    return good_mask, cutoff_mask, best_f[0]
