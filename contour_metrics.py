from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os


class contour_metrics:
    '''This class takes a thresholded image and uses Hu moments and other measures to estimate the probability that a '''

    def __init__(self, thresholded_img):
        thresholded_img = cv2.normalize(thresholded_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        self.h, self.w = thresholded_img.shape[0:2]
        self.crude_contours = cv2.findContours(thresholded_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] #must copy the thresholded_img because findContours overwrites its input
        self.smooth_contours = cv2.findContours(thresholded_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
        if len(self.crude_contours) != len(self.smooth_contours):
            print("oops - the two contour methods should give the same number of contour areas")
            exit
        else:
            self.n_contours = len(self.smooth_contours)


    #Methods for each contour
    def good_sized_area(self, i, cutoff=0.001):
        '''by default good sized areas are < 0.1% of the total image area'''
        return cv2.contourArea(self.smooth_contours[i]) > cutoff*self.h*self.w
    
    def contour_points(self, i):
        '''Return the number of vertices in the contour i, and in the same contour when smoothed'''
        return len(self.crude_contours[i]), len(self.smooth_contours[i])

    def approx_contour_points(self, i):
        return len(cv2.approxPolyDP(self.crude_contours[i],1,True)), len(cv2.approxPolyDP(self.smooth_contours[i],1,True))

    def contour_area(self, i):
        return cv2.contourArea(self.smooth_contours[i])

    def Hu(self, i):
        '''Return the Hu moments for contour i'''
        return cv2.HuMoments(cv2.moments(self.smooth_contours[i]))
        
    def all_params(self, i):
        return self.contour_points(i) + self.approx_contour_points(i) + (self.contour_area(i),) + tuple(self.Hu(i).flatten())

    @staticmethod
    def header():
        return "img-contour	crude.points	smooth.points	simp.crude.points	simp.smooth.points	area	hu1	hu2	hu3	hu4	hu5	hu6	hu7";

    def contour_image(self, i):
        '''Save to file the 7 Hu moments for each contour in each image for statistical analysis (e.g. for analysis to predict
        which are butterfly shaped). Also save images of each contour so we can look through and mark by hand which are the
        butterfly outlines'''
        roi = np.asarray(cv2.boundingRect(self.smooth_contours[i]))
        img = np.zeros((roi[3], roi[2],3), np.uint8)
        cv2.drawContours(img, [self.smooth_contours[i]-roi[0:2]], 0, (255,255,255), 2)
        return img

        
    def prob_butterfly(self, crude_contours, smooth_contours, statistical_model_to_use=0):
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

        values = np.zeros(self.n_contours)
        for i in range(self.n_contours):
            pts = self.contour_points(i)
            if pts[0]==1: #to avoid div by 0 in smoothness calc. This is a pointless contour anyway
                values[i]=0
            else:
                Hu = self.Hu(i)
                hu1 = Hu[0]
                hu2 = Hu[1]
                hu4 = Hu[3]
                if(Hu[5] < 1e-11):
                    log_hu6 = np.log(1e-11)
                else:
                    log_hu6 = np.log(Hu[5])
    
                if statistical_model_to_use == 0:
                    x = -82.70066 +417.23428*hu1 -862.08239*hu1**2 -0.31457*log_hu6 + 35.49917*np.log(pts[1])/np.log(pts[0])
                elif statistical_model_to_use == 1:
                    epsilon=1e-4
                    x = -334.561 -142.494*np.log(hu1+epsilon) -48.361*np.log(hu1+epsilon)**2 -46.982*np.log(hu4+epsilon) -2.762*np.log(hu4+epsilon)**2 + 41.824*np.log(pts[1])/np.log(pts[0])
                elif statistical_model_to_use == 3:
                    x = -86.185 + 710.382*hu1   -1369.683*hu1**2 -9.629*hu2 -7695.691*hu2**2 -4701.824*hu4
                values[i] = np.exp(x)/(1+np.exp(x))
            
        return values

    def find_butterfly(self, models = [0]):
        '''Find all contours in the thresholded image, and for each contour, use the Hu moments, plus an estimate of the proportion of the contour that consists of straight lines,
        as predictors of the probability that a contour represents a butterfly shape. 
        Returns 2 vectors, giving max probabilities & indices into the contour list, for each of models specified by the models parameter (usually just one). 
        The list of contours is the 3rd output vector. For details of the models available, see the function prob_butterfly()'''
        prob = []
        index = []
        
        for statistical_model_to_use in [0]:
            pr = self.prob_butterfly(self.crude_contours, self.smooth_contours, statistical_model_to_use)
            p_large = [pr[x] if self.good_sized_area(x) else 0 for x in range(len(pr))]
            prob.append(np.max(p_large))
            index.append(np.argmax(p_large))
        return prob, index, self.crude_contours

class contour_metrics_output:
    def __init__(self, contour_dir, filename):
        self.contour_dir = contour_dir
        self.writefile = open(os.path.join(contour_dir,filename), 'w')
        self.writefile.write("{}\n".format(contour_metrics.header()))

    def write(self, contour_metrics_obj, EoLobjectID, output_contour_pics=False, output_numpy_contour_coords=False):
        for i in range(contour_metrics_obj.n_contours):
            if contour_metrics_obj.good_sized_area(i):
                if output_contour_pics:
                    cv2.imwrite(os.path.join(self.contour_dir, "{}-{}.jpg".format(EoLobjectID, i)), contour_metrics_obj.contour_image(i))

                if output_numpy_contour_coords:
                    np.save(os.path.join(self.contour_dir, "{}-{}.npy".format(EoLobjectID, i)), contour_metrics_obj.smooth_contours[i])

                #write params to param file
                rowname = "{}-{}".format(EoLobjectID ,i)
                params = contour_metrics_obj.all_params(i)
                self.writefile.write("{}\n".format("	".join(map(str,(rowname,) + params))))
                
        self.writefile.flush()