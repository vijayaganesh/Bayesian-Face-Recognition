import cv2
import numpy as np
import glob
import os

class Train_Gray_Hist:
    pos_dir = os.path.abspath("../../Dataset/Training/positive/")
    neg_dir = os.path.abspath("../../Dataset/Training/negative/")
    output_dir = os.path.abspath("../../Output/Images/")
    
    def __init__(self):
        pos_files = glob.glob(self.pos_dir+'/*.jpg')
        neg_files = glob.glob(self.neg_dir+'/*.jpg')
        self.compute_likelihood(pos_files,'positive')
        self.compute_likelihood(neg_files,'negative')
    def compute_likelihood(self,files,data_type):
        img_array = []
        for file in files:
            x_array = self.processImage(file)
            img_array.append(x_array)
        mu,sigma,image_data = self.compute_stat(img_array)
        np.save(self.output_dir+"/gray_hist_image_data_"+data_type,image_data)
        np.save(self.output_dir+"/gray_hist_mu_"+data_type,mu)
<<<<<<< HEAD:src/training_gauss/train_gray_hist.py
        mu = np.round(mu.reshape((100,100,1))).astype(np.uint8)
#        if(np.min(sigma)<0):
#            sigma = sigma - np.min(sigma)
=======
        mu = np.round(mu.reshape((60,60,1))).astype(np.uint8)
        if(np.min(sigma)<0):
            sigma = sigma - np.min(sigma)
>>>>>>> parent of e6e1666b... Added Mixture of Gaussian Models:src/training/train_gray_hist.py
        sig = np.copy(sigma)
        sig *= (255.0/sig.max())
        sig = np.round(sig).reshape((60,60,1)).astype(np.uint8)
        # print(sig
        cv2.imwrite(self.output_dir+"/gray_hist_mu_"+data_type+".jpg",mu)
        cv2.imwrite(self.output_dir+"/gray_hist_sig_"+data_type+".jpg",sig)
    def compute_stat(self,pixel_array):
        np_pixel = np.array(pixel_array)
        mean = np.mean(np_pixel,axis = 0)
        cov = np.cov(np_pixel.T)
        cov_diag = np.diag(cov)
        print(np.mean(cov_diag))
        print(mean.shape,cov_diag.shape)
        return mean,cov_diag,np_pixel
    def processImage(self,image_file):
        image = cv2.imread(image_file)
        gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hist_eq = cv2.equalizeHist(gray_img)
        return hist_eq.ravel()
if __name__ == '__main__':
    Train_Gray_Hist()
