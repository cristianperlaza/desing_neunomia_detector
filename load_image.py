import pydicom as dicom
import numpy as np
from PIL import Image
import cv2


import preprocess_img
import load_model
from tensorflow.keras import backend as K
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


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

def grad_cam(array): 
    img = preprocess_img.preprocess(array)
    model = load_model.model()
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:,argmax]
    last_conv_layer = model.get_layer('conv10_thisone')
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)
    for filters in range(64):
        conv_layer_output_value[:,:,filters] *= pooled_grads_value[filters]
    #creating the heatmap
    heatmap = np.mean(conv_layer_output_value, axis = -1)
    heatmap = np.maximum(heatmap, 0)# ReLU
    heatmap /= np.max(heatmap)# normalize
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array , (512 , 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency,img2)  
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:,:,::-1]


