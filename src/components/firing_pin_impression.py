# Library Imports
import sys
import os

import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt
from itertools import chain

import hdbscan

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class Firing_pin_config:
    masked_final_image:str = os.path.join('Data','Masked_output_image')

class firing_pin_identification:
    def __init__(self):
        self.firing_pin = Firing_pin_config()

    def firing_pin_hough(self):
        try:
            # Read OTSU image. 
            img = cv2.imread("Data/Otsu.png", cv2.IMREAD_COLOR) 
            
            # Convert to grayscale. 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            
            # Blur using 3 * 3 kernel. 
            gray_blurred = cv2.blur(gray, (3, 3)) 
            
            # Apply Hough transform on the blurred image. 
            detected_circles = cv2.HoughCircles(gray_blurred,  
                            cv2.HOUGH_GRADIENT, 1, 20,  param1 = 50, 
                        param2 = 30, minRadius = 30, maxRadius = 50) 
            
            # Draw circles that are detected. 
            if detected_circles is not None: 
            
                # Convert the circle parameters a, b and r to integers. 
                detected_circles = np.uint16(np.around(detected_circles)) 
            
                for pt in detected_circles[0, :]: 
                    a, b, r = pt[0], pt[1], pt[2] 
            
                    # Draw the circumference of the circle. 
                    cv2.circle(img, (a, b), r+20, (0, 255, 0), 2) 
            
                    # Draw a small circle (of radius 1) to show the center. 
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
                    cv2.imwrite("Data/Hough_circle.png", img)

            # Data points matching hough circle and firing pin points are identified.

            # Circle points
            new_img = np.zeros((gray.shape[0],gray.shape[1]),dtype='int8')

            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 

            # Draw the circumference of the circle. 
            cv2.circle(new_img, (a, b), r+150, (255, 0, 0), 0) 

            # Dataframe for circle points
            # Indexes for array 
            indices = np.dstack(np.indices(new_img.shape))
            flatten_array = list(chain.from_iterable(indices))

            # Dataframe with pixel values and positions
            df_circle = pd.DataFrame(flatten_array,columns=['rows','columns'])
            df_circle['values'] = new_img.flatten()
            new = df_circle[df_circle['values']!=0] # Circle points

            # Greyscale points
            gray_img = np.copy(gray)

            # Dataframe for gray image points
            # Indexes for array 
            indices = np.dstack(np.indices(gray_img.shape))
            flatten_array = list(chain.from_iterable(indices))

            # Dataframe with pixel values and positions
            df_gray = pd.DataFrame(flatten_array,columns=['rows','columns'])
            df_gray['values_gray'] = gray_img.flatten()

            # Matching points between cicle and FP points in original image
            merged_df = pd.merge(df_circle, df_gray, on=["rows", "columns"])

            # Matching points
            merged_df["patch"] = np.where((merged_df['values'] == 127) & (merged_df['values_gray'] <100), 255, 0)
            
            # Save patch
            patch = np.array(merged_df['patch'])
            patch = np.reshape(patch,new_img.shape)
            patch = np.uint8(patch)
            cv2.imwrite('Data/patch.png',patch)

            # Finding midpoint of patch using edge detection

            #ORB
            # Oriented FAST and Rotated BRIEF
            img = cv2.imread("Data/patch.png", 0)

            orb = cv2.ORB_create(1)

            kp, des = orb.detectAndCompute(img, None)

            # Draw keypoints location
            img2 = cv2.drawKeypoints(img, kp, None, flags=None)

            # Point
            import math
            for x in kp:
                i = math.ceil(x.pt[0])
                j = math.ceil(x.pt[1])

            # Arrow for FP drag
            new_img = cv2.imread("Data/Final.png")

            cv2.circle(new_img, (detected_circles[0][0][0],detected_circles[0][0][1]), 15, (255,0,0), 3) 
            cv2.arrowedLine(new_img, (detected_circles[0][0][0],detected_circles[0][0][1]), (i,j), (255,0,0), 3, 5, 0, 0.1)

            cv2.imwrite('Data/Final_image.png',new_img)


        except Exception as e:
            raise CustomException(e,sys)
