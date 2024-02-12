# Library Imports
import sys
import os

import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt

import hdbscan

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class ModeltrainerConfig:
    output_file_path:str = os.path.join('Data','BF_Output.png')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def model(self,df,pic_shape):
        try:
            # Model training to find clusters
            # HDBSCAN algorithm

            df_use = df[(df['values']==255)] # To get the breech-face impression part of image.

            clusterer = hdbscan.HDBSCAN(min_cluster_size=200, gen_min_span_tree=True, min_samples=4)
            clusterer.fit(df_use.iloc[:,:2])

            # Merging clusters to Dataframe
            df_use['HDBSCAN_labels']=clusterer.labels_ 
            fin_df = pd.merge(df, df_use, on=["rows", "columns"], how='left')
            fin_df['HDBSCAN_labels'] = fin_df['HDBSCAN_labels'].fillna(-2).astype(int)
        
            # Take data points from the breech-face cluster and mask original image.

            test = fin_df.groupby(by='HDBSCAN_labels').count()
            test.drop([-2], axis=0, inplace=True)

            test = test.reset_index().sort_values(by='values_y',ascending=False)
            label = test['HDBSCAN_labels'].iloc[0]

            fin_df["mask"] = np.where((fin_df["HDBSCAN_labels"] == label), 255, 0)

            #Visualize mask area
            arr = fin_df['mask'].array.reshape(pic_shape)
            exp_arr = np.expand_dims(arr, axis=(-1))
            sam = np.uint8(exp_arr)

            # Mask original image
            sam_img = cv2.imread("Data/Picture.png")

            pixels = fin_df[fin_df['mask'] == 255]
            lis1 = list(pixels['rows'])
            lis2 = list(pixels['columns'])

            for i, j in zip(lis1, lis2):
                sam_img[i][j] = [0,0,255]
                
            cv2.imwrite(self.model_trainer_config.output_file_path,sam_img)

        except Exception as e:
            raise CustomException(e,sys)
        
