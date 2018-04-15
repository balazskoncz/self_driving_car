# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:26:42 2018

@author: dev
"""

import matplotlib.image as mpimg
import numpy as np
import cv2
import random

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    
    return image

def crop(image):
    return image[60:-25, :, :]

def resize(image):
    return cv2.resize(image, (200, 66), cv2.INTER_AREA)


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def get_trainer_image(centre, left, right, angle):
    selection = np.random.choice(3)
    
    if selection == 0:
        return mpimg.imread(left), angle +0.2
    elif selection == 1:
        return mpimg.imread(right), angle-0.2
    return mpimg.imread(centre), angle
    
def flip(image, angle, probability):
    
    if np.random.rand() < probability:
        image = cv2.flip(image, 1)
        angle = -angle
        
    return image, angle
    
def shift(image, angle, x_range, y_range, threshold):
    x_prime = x_range * (np.random.rand() -threshold)
    y_prime = y_range * (np.random.rand() -threshold)
    
    angle += x_prime * 0.002
    
    transformer_matrix = np.float32([[1, 0, x_prime], [0, 1, y_prime]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, transformer_matrix, (width, height))
    
    return image, angle

def adjust_brightness(image, threshold):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - threshold)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_shadow(image):
    x1, y1 = 200 * np.random.rand(), 0
    x2, y2 = 200 * np.random.rand(), 66
    xm, ym = np.mgrid[0:66, 0:200]

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def augment(centre, left, right, angle):
    
    image, angle = get_trainer_image(centre, left, right, angle)
    image, angle = flip(image, angle, 0.5)
    image, angle = shift(image, angle, 100, 10, 0.5)
    image = adjust_brightness(image, 0.5)
    image = add_shadow(image)
    
    return image, angle

    
def training_generator(features, labels, batch_size, training):
    number_of_features = len(features)
    feature_batches = np.zeros((batch_size, 66, 200, 3))
    label_batches = np.zeros((batch_size,1))
    
    while True:
        for index in range(batch_size):
            selection_index = random.randint(0, number_of_features-1)
            centre, left, right = features[selection_index]
            
            if training and np.random.rand() < 0.6:
                train_image, train_angle = augment(centre, left, right, labels[index])
                feature_batches[index] = preprocess(train_image)
                label_batches[index] = train_angle
            else:
                feature_batches[index] = preprocess(mpimg.imread(centre))
                label_batches[index] = labels[index]
                
        yield feature_batches, label_batches
        
    
def get_images(features):
    images = []
    for sample in features:
        centre, left, right = sample
        
        centre_image = mpimg.imread(centre)
        centre_image = preprocess(centre_image)
        
        left_image = mpimg.imread(left)
        left_image = preprocess(left_image)
        
        right_image = mpimg.imread(right)
        right_image = preprocess(right_image)
        
        image_sample = [centre_image, left_image, right_image]
        
        images.append(image_sample[0])
        
    return np.array(images)