import cv2
import numpy as np
import glob
import os,sys
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from utilities.Utilities import Utilities

class Predict:
    likelihood_dir = os.path.abspath('../../Output/Numpy/mog/')
    pos_test_dir = os.path.abspath("../../Dataset/Test/positive/")
    neg_test_dir = os.path.abspath("../../Dataset/Test/negative/")
    print(pos_test_dir)
    reduced_dimensions = 50
    def __init__(self,type):
            mu0,cov0,mu1,cov1,theta,shape =self.load_model(type)
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
#                print("Correct/Total Predictions: "+repr((correct_pred,len(prediction))))
            print("Accuracy: "+repr(float(correct_pred)*100/len(prediction)))
            print(Utilities.performance_Metrics(prediction,self.ground_truth))
            fp,tp,_ = roc_curve(self.ground_truth,prediction)   
            area_under_curve = auc(fp, tp)
            plt.plot(fp, tp, 'b',label='AUC = %0.2f'% area_under_curve)
            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.1,1.1])
            plt.ylim([-0.1,1.1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show()

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
        mu1 = np.load(os.path.join(self.likelihood_dir,type+"_mu.npy"))
        cov1 = np.load(os.path.join(self.likelihood_dir,type+"_cov.npy"))
        mu0 = np.load(os.path.join(self.likelihood_dir,type+"_mu_neg.npy"))
        cov0 = np.load(os.path.join(self.likelihood_dir,type+"_cov_neg.npy"))
        theta = np.load(os.path.join(self.likelihood_dir,type+"_theta.npy"))
        shape = mu0.shape
        return mu0,cov0,mu1,cov1,theta,shape
    def pdf_nd(self,mu,cov_det,cov_inv,x):
        prod = -0.5 * np.matmul(np.matmul(x-mu,cov_inv),np.transpose(x-mu))
        prob = (1/np.sqrt(2*np.pi*cov_det))*np.exp(prod)
        return prob
    def compute_posterior(self,lh_0,lh_1,prior=0.5):
        return (lh_1*prior/(lh_0*prior+lh_1*prior))

if __name__ == '__main__':
    Predict("hsv")
