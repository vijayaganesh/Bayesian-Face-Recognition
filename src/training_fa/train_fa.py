# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:32:45 2018

@author: VijayaGanesh Mohan
@email: vmohan2@ncsu.edu

"""

import numpy as np
from factor_analyzer import Factor_Analyzer
import cv2
import glob
import os
class train_fa:
    pos_dir = os.path.abspath("../../Dataset/Training/positive/")
    neg_dir = os.path.abspath("../../Dataset/Training/negative/")
    output_image_dir = os.path.abspath("../../Output/Images/fa/")
    output_numpy_dir = os.path.abspath("../../Output/Numpy/fa/")
    reduced_dimensions = 50
    def __init__(self,file_type,factors):
        pos_files = glob.glob(self.pos_dir+'/*.jpg')
        neg_files = glob.glob(self.neg_dir+'/*.jpg')
        files = glob.glob(self.output_image_dir+'/'+file_type+'*.jpg')
        for f in files:
            os.remove(f)
        files = glob.glob(self.output_numpy_dir+'/'+file_type+'*')
        for f in files:
            os.remove(f)
        self.compute_likelihood(pos_files,'positive',file_type,factors)
        self.compute_likelihood(neg_files,'negative',file_type,factors)
    def compute_likelihood(self,files,data_type,file_type,k):
        img_array = []
        for file in files:
            x_array = self.processImage(file,file_type)
            img_array.append(x_array)
        self.compute_stat(img_array,data_type,file_type,k)
    def processImage(self,image_file,file_type):
        if(file_type == 'rgb'):
            image = cv2.imread(image_file)
        elif(file_type == 'hsv'):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        else:
            image = cv2.imread(image_file)
            gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(gray_img)
        return image.ravel()
    def compute_stat(self,img_array,data_type,file_type,factors):
        img_array = np.array(img_array)
        if data_type == 'positive':
            model = Factor_Analyzer(img_array,factors)
            model.exp_max(iterations = 4)
            for i in range(0,factors):
                mu_1 = model.mu + 2*(model.phi[:,i]/np.mean(model.phi[:,i]))
                mu_2 = model.mu - 2*(model.phi[:,i]/np.mean(model.phi[:,i]))
                print(model.phi[:,i])
                if(file_type == "rgb"):
                    mu_1 = np.round(mu_1.reshape((100,100,3))).astype(np.uint8)
                    mu_2 = np.round(mu_2.reshape((100,100,3))).astype(np.uint8)
                elif file_type == "hsv":
                    mu_1 = np.round(mu_1.reshape((100,100,3))).astype(np.uint8)
                    mu_1 = cv2.cvtColor(mu_1,cv2.COLOR_HSV2BGR)
                    mu_2 = np.round(mu_2.reshape((100,100,3))).astype(np.uint8)
                    mu_2 = cv2.cvtColor(mu_2,cv2.COLOR_HSV2BGR)
                else:
                    mu_1 = np.round(mu_1.reshape((100,100,1))).astype(np.uint8)
                    mu_2 = np.round(mu_2.reshape((100,100,1))).astype(np.uint8)
#                print(np.max(mu))
                cv2.imshow('mean',mu_1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow('mean',mu_2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
#                    print(self.output_image_dir+file_type+"+"_mog_"+repr(i)
                cv2.imwrite(self.output_image_dir+"/"+file_type+"_fa_mu_plus_phi_"+repr(i+1)+".jpg",mu_1)
                cv2.imwrite(self.output_image_dir+"/"+file_type+"_fa_mu_minus_phi_"+repr(i+1)+".jpg",mu_2)
            np.save(self.output_numpy_dir+"/"+file_type+"_mu",model.mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_cov",model.cov)
            np.save(self.output_numpy_dir+"/"+file_type+"_phi",model.phi)
        else:
            mu = np.mean(img_array,axis = 0)
            if(file_type == "rgb"):
                mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
            elif file_type == "hsv":
                mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
                mu = cv2.cvtColor(mu,cv2.COLOR_HSV2BGR)
            else:
                mu = np.round(mu.reshape((100,100,1))).astype(np.uint8)
            cv2.imwrite(self.output_image_dir+"/"+file_type+"_mog_neg.jpg",mu)
            mu = np.mean(img_array,axis = 0)
            cov = np.cov(img_array.T)
            np.save(self.output_numpy_dir+"/"+file_type+"_mu_neg",mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_cov_neg",cov)
            
if __name__ == '__main__':
    train_fa('rgb',4)