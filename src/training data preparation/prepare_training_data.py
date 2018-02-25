import cv2
import random
import os
seed =5

class Extract_Data:
    annot_file = os.path.abspath("../../Raw Data/FDDB-folds/FDDB-fold-")
    annot_num = 10
    test_data = 100
    training_positive_output_dir = os.path.abspath("../../Dataset/Training/positive/")
    training_negative_output_dir = os.path.abspath("../../Dataset/Training/negative/")
    test_positive_output_dir = os.path.abspath("../../Dataset/Test/positive/")
    test_negative_output_dir = os.path.abspath("../../Dataset/Test/negative/")
    image_dir = os.path.abspath("../../Raw Data/Images/")
    face_count = 0
    image_count = 0
    def __init__(self):
        while(self.annot_num != 0):
            self.parse_annotate_file(self.annot_file+"{:0>2d}".format(self.annot_num))
            self.annot_num -= 1
    def parse_image(self,img_file,face_list):
        image = cv2.imread(img_file)
        h,w,_ = image.shape
        for face in face_list:
            coords = face.split(" ")
            width = int(round(max(float(coords[0]),float(coords[1]))))
            center_x,center_y = int(round(float(coords[3]))),int(round(abs(float(coords[4]))))
            rand_x,rand_y = random.randrange(width,w),random.randrange(width,h)
            x,y = max(0,center_x - width) , max(0,center_y - width)
            cropped_img = cv2.resize(image[y:y+width*2, x:x+width*2],(100,100),interpolation = cv2.INTER_AREA)
            cropped_neg_image = cv2.resize(image[rand_y:rand_y+width*2, rand_x:rand_x+width*2],(100,100),interpolation = cv2.INTER_AREA)
            self.face_count += 1
            if(self.face_count<=self.test_data):
                cv2.imwrite(self.test_positive_output_dir+'/'+repr(self.face_count)+'.jpg',cropped_img)
                cv2.imwrite(self.test_negative_output_dir+'/'+repr(self.face_count)+'.jpg',cropped_neg_image)
            else:
                cv2.imwrite(self.training_positive_output_dir+'/'+repr(self.face_count-self.test_data)+'.jpg',cropped_img)
                cv2.imwrite(self.training_negative_output_dir+'/'+repr(self.face_count-self.test_data)+'.jpg',cropped_neg_image)
            print("Images Read/Faces Detected = ("+repr(self.image_count)+"/"+repr(self.face_count)+")")
    def parse_annotate_file(self,file_name):
        id_count = 0
        img_file = ""
        with open(file_name+'-ellipseList.txt','r') as f:
            for line in f:
                if(id_count == 0 and img_file ==""):
                    img_file = self.image_dir+'/'+line.replace("\n","")+'.jpg'
                    self.image_count += 1
                    face_coords = list()
                elif(id_count ==0 and img_file != ""):
                    id_count = int(line)
                else:
                    face_coords.append(line)
                    id_count -= 1
                    if(id_count == 0 ):
                        self.parse_image(img_file,face_coords)
                        img_file =""
if __name__ == "__main__":
    Extract_Data()
