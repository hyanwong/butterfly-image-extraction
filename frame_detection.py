from __future__ import division
from __future__ import print_function
import cv2
import numpy as np

def using_rectangular_contour(img, min_prop_picture_in_frame, verbose=False):
    '''Detect if picture is in a frame, by detecting rectangles and crop if any contain >75% of the image.
    This code does not seem to work consistently (e.g. on eol.org/data_objects/17762955): it probably needs tweaking'''
    h, w = img.shape[:2]
    epsilon = (h+w)/2 * 0.01 # allow lines in rectange detection to deviate about 1% of the size of the image
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

def using_floodfill(img, edge_fraction=0.1, line_length = 0.8, maxLineGap=10, horiz_vert_max_gradient = 1/40, verbose=False):
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
                    if verbose:
                        print("got x min")
                elif (best[2] is None or (x_pos < best[2])) and (x_pos > 0.9*w): #in rightmost 10% of image 
                    best[2] = np.max(p[[0,2]])
                    if verbose:
                        print("got x max")
    horiz_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/2, threshold =4, minLineLength=w* line_length, maxLineGap= maxLineGap)
    if horiz_lines is not None:
        for l in range(horiz_lines.shape[1]):
            p = horiz_lines[0,l,:]
            if (p[0] != p[2]) and abs((p[1]-p[3])/(p[0]-p[2]))< horiz_vert_max_gradient: #vert has gradient 1 in 40 or shallower
                y_pos = np.mean(p[[1,3]])
                if   (best[1] is None or (y_pos > best[1])) and (y_pos < 0.1*h): #in topmost 10% of image 
                    best[1] = np.min(p[[1,3]])
                    if verbose:
                        print("got y min")
                elif (best[3] is None or (y_pos < best[3])) and (y_pos > 0.9*h): #in bottommost 10% of image 
                    best[3] = np.max(p[[1,3]])
                    if verbose:
                        print("got y max")


    if all(i is not None for i in best):
        if verbose:
            print("Cropped")
        return best[0], best[1], w-best[2], h-best[3]
    else:
        return 0,0,0,0
