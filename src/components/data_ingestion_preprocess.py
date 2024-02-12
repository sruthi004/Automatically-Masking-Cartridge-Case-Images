# Library imports
import pandas as pd
import numpy as np

from glob import glob
import cv2
import matplotlib.pylab as plt
import seaborn as sns

from collections import Counter

from itertools import chain
from dataclasses import dataclass
import os
import sys

from model_training_masking import Modeltrainer
from firing_pin_impression import firing_pin_identification

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    picture:str = os.path.join('Data','Picture.png')
    otsu:str = os.path.join('Data','Otsu.png')
    

class Data_ingestion_process:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Import image data
            pic = glob("Data/Picture1.png")
            img = cv2.imread(pic[0])

            # Add to Data folder
            print(type(img))
            cv2.imwrite(self.ingestion_config.picture, img)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def preprocess(self):
        try:
            # Converting image to grayscale
            img = cv2.imread(self.ingestion_config.picture)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 

            # OTSU for segmentation
            # Creating a CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  
            clahe_img = clahe.apply(img_gray)

            plt.hist(clahe_img.flat, bins =100, range=(0,255))

            # Otsu's thresholding, automatically finds the threshold point. 
            ret1,th1 = cv2.threshold(clahe_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(self.ingestion_config.otsu,th1)

            # Creating a dataframe of pixel data for model training
            fin_pic = np.copy(th1)

            # Indexes from array 
            indices = np.dstack(np.indices(fin_pic.shape))
            flatten_array = list(chain.from_iterable(indices))

            # Dataframe with pixel values and positions
            df = pd.DataFrame(flatten_array,columns=['rows','columns'])
            df['values'] = th1.flatten()
            return(df, fin_pic.shape)

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = Data_ingestion_process()
    obj.initiate_data_ingestion()
    df,pic_shape = obj.preprocess()
   
    model = Modeltrainer()
    model.model(df,pic_shape)

    firing = firing_pin_identification()
    firing.firing_pin_hough()
    