import cv2
import numpy as np
import glob
import os

class Train_HSV:
    pos_dir = os.path.abspath("../../Dataset/Training/positive/")
    neg_dir = os.path.abspath("../../Dataset/Training/negative/")
    output_dir = os.path.abspath("../../Output/Images/")
    def __init__(self):
        pos_files = glob.glob(self.pos_dir+'/*.jpg')
        neg_files = glob.glob(self.neg_dir+'/*.jpg')
        self.compute_likelihood(pos_files,'positive')
        self.compute_likelihood(neg_files,'negative')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def compute_likelihood(self,files,data_type):
        img_array = []
        for file in files:
            x_array = self.processImage(file)
            img_array.append(x_array)
        mu,sigma,image_data = self.compute_stat(img_array)
        np.save(self.output_dir+"/hsv_image_data_"+data_type,image_data)
        np.save(self.output_dir+"/hsv_mu_"+data_type,mu)
        mu = np.round(mu.reshape((100,100,3))).astype(np.uint8)
        if(np.min(sigma)<0):
            sigma = sigma - np.min(sigma)
        sig = np.copy(sigma)
        sig *= (255.0/sig.max())
        sig = np.round(sig).reshape((100,100,3)).astype(np.uint8)
        mu_image = cv2.cvtColor(mu,cv2.COLOR_HSV2BGR)
        sig_image = cv2.cvtColor(sig,cv2.COLOR_HSV2BGR)
        cv2.imwrite(self.output_dir+"/hsv_mu_"+data_type+".jpg",mu_image)
        cv2.imwrite(self.output_dir+"/hsv_sig_"+data_type+".jpg",sig_image)
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
        image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        return image_hsv.ravel()
if __name__ == '__main__':
    Train_HSV()
