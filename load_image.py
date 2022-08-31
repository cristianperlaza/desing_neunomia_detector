import pydicom as dicom
import numpy as np
from PIL import Image
import cv2

def read_dicom_file(path):    
    img = dicom.read_file(path)    
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)  
    img2 = img_array.astype(float) 
    img2 = (np.maximum(img2,0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)        
    return img_RGB, img2show

def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array) 
    img2 = img_array.astype(float) 
    img2 = (np.maximum(img2,0) / img2.max()) * 255.
    img2 = np.uint8(img2)
    return img2, img2show 


def preprocess(array):
     array = cv2.resize(array , (512 , 512))
     array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
     array = clahe.apply(array)
     array = array/255
     array = np.expand_dims(array,axis=-1)
     array = np.expand_dims(array,axis=0)
     return array


