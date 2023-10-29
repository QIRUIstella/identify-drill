# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 02:25:28 2022

@author: huife
"""

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import sys

# names of result classes
class_names = ['2.0 mm x 26 mm', '2.0 mm x 28 mm', '2.8 mm x 22 mm', '3.5 mm x 19 mm', '3.5 mm x 22 mm', 
               '3.5 mm x 28 mm', '3.5 mm x 30 mm', '4.2 mm x 22 mm', '4.2 mm x 30 mm']

# crop original image, only keep middle left area
def crop_img(img_path):
    im = Image.open(img_path)
    width, height = im.size
    
    left = 0
    top = height / 4
    right = width / 2
    bottom = 3 * height / 4
    
    result = im.crop((left, top, right, bottom))
    return result

# run my ML model
def run_model(img_path):
    test_img = img_path
    test_img = crop_img(test_img)
    
    img_array = tf.keras.utils.img_to_array(test_img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    # load my model
    my_model = keras.models.load_model("my_model")
    # make prediction
    predictions = my_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # return result class
    return class_names[np.argmax(score)]
    

if __name__ == "__main__":
    
    img = input("Please Enter Image Path: \n")
    print("This Image Belongs to: \n", run_model(img))