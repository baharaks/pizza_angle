#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:26:07 2024

@author: becky
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sqlite3
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import math 
from skimage import io, transform
from skimage.transform import resize, rotate
from torchvision import transforms, utils
  

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.new_h, self.new_w = (150, 150)

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        h, w = image.size

        img = image.resize((self.new_h, self.new_w))
        keypoints = keypoints * [self.new_w / w, self.new_h / h]
        
        return {'image': img, 'keypoints': keypoints}

class Rotate(object):
    """Rotate an image and update the coordinates of the pizza slice."""
    def __init__(self, angle):
        self.angle = angle
        self.new_h, self.new_w = (150, 150)
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        
        # Rotate image
        image = image.rotate(self.angle, Image.NEAREST, expand = 1)
        # Update coordinates
        w, h = image.size
        cx, cy = w / 2, h / 2
        rad_angle = math.radians(self.angle)
        
        def rotate_point(x, y):
            x_new = (x - cx) * math.cos(rad_angle) + (y - cy) * math.sin(rad_angle) + cx
            y_new = -(x - cx) * math.sin(rad_angle) + (y - cy) * math.cos(rad_angle) + cy
            return x_new, y_new
        
        pizza_tip_x, pizza_tip_y = keypoints[0][0] 
        behind_tip_x, behind_tip_y = keypoints[0][1]
        pizza_tip_x_new, pizza_tip_y_new = rotate_point(pizza_tip_x, pizza_tip_y)
        behind_tip_x_new, behind_tip_y_new = rotate_point(behind_tip_x, behind_tip_y)

        # keypoints = (pizza_tip_x_new, pizza_tip_y_new, behind_tip_x_new, behind_tip_y_new)
        keypoints = [(pizza_tip_x_new, pizza_tip_y_new), (behind_tip_x_new, behind_tip_y_new)]
        
                
        keypoints = np.array([keypoints], dtype=float)
        h, w = image.size
        img = image.resize((self.new_h, self.new_w))
        
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        keypoints = keypoints * [self.new_w / w, self.new_h / h]
        
        return {'image': img, 'keypoints': keypoints}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W        
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}
    
class PizzaSliceDataset(Dataset):
    def __init__(self, dir_path, list_image, list_keypoints, transform=None):
        self.list_image = list_image
        self.list_keypoints = list_keypoints
        self.dir_path = dir_path
        self.transform = transform        

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, idx):
        
        image_name = self.list_image[idx]
        pizza_tip_x, pizza_tip_y, behind_tip_x, behind_tip_y = self.list_keypoints[idx]
        image_path = os.path.join(self.dir_path, image_name)
        
        image = Image.open(image_path).convert('RGB')
        img = image.resize((150, 150))
        keypoints = [(pizza_tip_x, pizza_tip_y), (behind_tip_x, behind_tip_y)]
        
                
        keypoints = np.array([keypoints], dtype=float) 
        
        sample = {'image': image , 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
