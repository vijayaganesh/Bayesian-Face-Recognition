# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:05:29 2018

@author: VijayaGanesh Mohan
@email: vmohan2@ncsu.edu

"""
import numpy as np
from T_Dist import T_Dist
import cv2
import glob
from sklearn.decomposition import PCA
import os
class train_T_Dist:
    pos_dir = os.path.abspath("../../Dataset/Training/positive/")
    neg_dir = os.path.abspath("../../Dataset/Training/negative/")
    output_image_dir = os.path.abspath("../../Output/Images/t_dist/")
    output_numpy_dir = os.path.abspath("../../Output/Numpy/t_dist/")
    reduced_dimensions = 30
    def __init__(self,file_type):
        pos_files = glob.glob(self.pos_dir+'/*.jpg')
        neg_files = glob.glob(self.neg_dir+'/*.jpg')
        files = glob.glob(self.output_image_dir+'/'+file_type+'*.jpg')
        for f in files:
            os.remove(f)
        files = glob.glob(self.output_numpy_dir+'/'+file_type+'*')
        for f in files:
            os.remove(f)
        self.compute_likelihood(pos_files,'positive',file_type)
        self.compute_likelihood(neg_files,'negative',file_type)
    def compute_likelihood(self,files,data_type,file_type):
        img_array = []
        for file in files:
            x_array = self.processImage(file,file_type)
            img_array.append(x_array)
        self.compute_stat(img_array,data_type,file_type)
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
    def compute_stat(self,img_array,data_type,file_type):
        img_array = np.array(img_array)
        pca = PCA(n_components=self.reduced_dimensions,svd_solver='randomized').fit(img_array)
        pca_array = pca.transform(img_array)
        if data_type == 'positive':
            model = T_Dist(pca_array)
            model.exp_max(tolerance = 0.1)
            mu = pca.inverse_transform(model.mu)
            if(file_type == "rgb"):
                mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
            elif file_type == "hsv":
                mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
                mu = cv2.cvtColor(mu,cv2.COLOR_HSV2BGR)
            else:
                mu = np.round(mu.reshape((100,100,1))).astype(np.uint8)
#                print(np.max(mu))
            print(model.nu)
            cv2.imshow('mean',mu)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#                    print(self.output_image_dir+file_type+"+"_mog_"+repr(i)
            cv2.imwrite(self.output_image_dir+"/"+file_type+"_t_dist.jpg",mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_mu",model.mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_cov",model.cov)
            np.save(self.output_numpy_dir+"/"+file_type+"_nu",model.nu)
        else:
            mu = np.mean(img_array,axis = 0)
            if(file_type == "rgb"):
                mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
            elif file_type == "hsv":
                mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
                mu = cv2.cvtColor(mu,cv2.COLOR_HSV2BGR)
            else:
                mu = np.round(mu.reshape((100,100,1))).astype(np.uint8)
            cv2.imwrite(self.output_image_dir+"/"+file_type+"_t_dist_neg.jpg",mu)
            pca_mu = np.mean(pca_array,axis = 0)
            cov = np.cov(pca_array.T)
            np.save(self.output_numpy_dir+"/"+file_type+"_mu_neg",pca_mu)
            np.save(self.output_numpy_dir+"/"+file_type+"_cov_neg",cov)
            
if __name__ == '__main__':
    train_T_Dist('gray_hist')
            
        
        
        
        
        
        
        
        
        
        
        