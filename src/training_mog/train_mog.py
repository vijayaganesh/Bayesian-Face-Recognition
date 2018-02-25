# -*- coding: utf-8 -*-

"""
Created on Fri Feb 23 17:57:13 2018

@author: VijayaGanesh Mohan
"""
import numpy as np
from mog import MoG
import cv2
import glob
from sklearn.decomposition import PCA
import os
class train_mog:
    pos_dir = os.path.abspath("../../Dataset/Training/positive/")
    neg_dir = os.path.abspath("../../Dataset/Training/negative/")
    output_image_dir = os.path.abspath("../../Output/Images/mog/")
    output_numpy_dir = os.path.abspath("../../Output/Numpy/mog/")
    reduced_dimensions = 50
    def __init__(self,file_type,k):
        pos_files = glob.glob(self.pos_dir+'/*.jpg')
        neg_files = glob.glob(self.neg_dir+'/*.jpg')
        files = glob.glob(self.output_image_dir+'/'+file_type+'*.jpg')
        for f in files:
            os.remove(f)
        files = glob.glob(self.output_numpy_dir+'/'+file_type+'*')
        for f in files:
            os.remove(f)
        self.compute_likelihood(pos_files,'positive',file_type,k)
        self.compute_likelihood(neg_files,'negative',file_type,k)
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
    def compute_stat(self,img_array,data_type,file_type,k):
        img_array = np.array(img_array)
        pca = PCA(n_components=self.reduced_dimensions,svd_solver='randomized').fit(img_array)
        pca_array = pca.transform(img_array)
        if data_type == 'positive':
            model = MoG(pca_array,k)
            model.exp_max(tolerance = 1e-18)
            for i in range(0,k):
                mu = pca.inverse_transform(model.mu[i])
                if(file_type == "rgb"):
                    mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
                elif file_type == "hsv":
                    mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
                    mu = cv2.cvtColor(mu,cv2.COLOR_HSV2BGR)
                else:
                    mu = np.round(mu.reshape((100,100,1))).astype(np.uint8)
#                print(np.max(mu))
                cv2.imshow('mean',mu)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
#                    print(self.output_image_dir+file_type+"+"_mog_"+repr(i)
                cv2.imwrite(self.output_image_dir+"/"+file_type+"_mog_"+repr(i+1)+".jpg",mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_mu",model.mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_cov",model.cov)
            np.save(self.output_numpy_dir+"/"+file_type+"_theta",model.theta)
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
            pca_mu = np.mean(pca_array,axis = 0)
            cov = np.cov(pca_array.T)
            np.save(self.output_numpy_dir+"/"+file_type+"_mu_neg",pca_mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_cov_neg",cov)
            
if __name__ == '__main__':
    train_mog('hsv',15)
            
        
        
        
        
        
        
        
        
        
        
        