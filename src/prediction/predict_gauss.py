import cv2
import numpy as np
import glob
import os
from sklearn.decomposition import PCA

class Predict:
    likelihood_dir = os.path.abspath('../../Output/Images/')
    pos_test_dir = os.path.abspath("../../Dataset/Test/positive/")
    neg_test_dir = os.path.abspath("../../Dataset/Test/negative/")
    output_dir = os.path.abspath("../../Output/")
    print(pos_test_dir)
    reduced_dimensions = 50
    def __init__(self,type):
            mu0,image0,mu1,image1,shape =self.load_model(type)
            image0,image1 = self.reduce_dimensions(image0),self.reduce_dimensions(image1)
            mu0,mu1 = np.mean(image0,axis = 0),np.mean(image1,axis = 0)
            print(mu0)
            cov0 = np.cov(image0.T)
            cov1 = np.cov(image1.T)
            cov0_det,cov0_inv =  np.linalg.det(cov0),np.linalg.inv(cov0)
            cov1_det,cov1_inv =  np.linalg.det(cov1),np.linalg.inv(cov1)
            self.ground_truth = list()
            test_images = self.load_test_data(type)
            test_pca = self.reduce_dimensions(test_images)
            prediction = list()
            correct_pred = 0
            for x,pred in zip(test_pca,self.ground_truth):
                lh_0 = self.pdf_nd(mu0,cov0_det,cov0_inv,x)
                lh_1 = self.pdf_nd(mu1,cov1_det,cov1_inv,x)
                prob = self.compute_posterior(lh_0,lh_1)
                if(prob>=0.5):
                    prob = 1
                else:
                    prob = 0
                if(prob == pred):
                    correct_pred += 1
                prediction.append(prob)
                print("Correct/Total Predictions: "+repr((correct_pred,len(prediction))))
            print("Accuracy: "+repr(float(correct_pred)*100/len(prediction)))

    def reduce_dimensions(self,image_array):
        pca = PCA(n_components=self.reduced_dimensions,svd_solver='randomized').fit(image_array)
        print(image_array.shape)
        return pca.transform(image_array)

    def load_test_data(self,type):
        pos_files = glob.glob(self.pos_test_dir+'/*.jpg')
        neg_files = glob.glob(self.neg_test_dir+'/*.jpg')
        test_images = []
        file_count = 0
        for file in pos_files:
            test_images.append(self.processImage(file,type))
            self.ground_truth.append(1)
            file_count += 1
        for file in neg_files:
            test_images.append(self.processImage(file,type))
            self.ground_truth.append(0)
        return np.array(test_images)

    def processImage(self,image_file,type):
        image = cv2.imread(image_file)
        if(type == "hsv"):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        elif(type == "gray_hist"):
            image = cv2.equalizeHist(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        return image.ravel()

    def load_model(self,type):
        mu1 = np.load(os.path.join(self.likelihood_dir,type+"_mu_positive.npy"))
        image1 = np.load(os.path.join(self.likelihood_dir,type+"_image_data_positive.npy"))
        mu0 = np.load(os.path.join(self.likelihood_dir,type+"_mu_negative.npy"))
        image0 = np.load(os.path.join(self.likelihood_dir,type+"_image_data_negative.npy"))
        shape = mu0.shape
        return mu0,image0,mu1,image1,shape
    def pdf_nd(self,mu,cov_det,cov_inv,x):
        prod = -0.5 * np.matmul(np.matmul(x-mu,cov_inv),np.transpose(x-mu))
        prob = (1/np.sqrt(2*np.pi*cov_det))*np.exp(prod)
        return prob
    def compute_posterior(self,lh_0,lh_1,prior=0.5):
        return (lh_1*prior/(lh_0*prior+lh_1*prior))

if __name__ == '__main__':
    Predict("rgb")
