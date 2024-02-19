#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:18:35 2024

@author: becky
"""

import torch
import numpy as np
import math
import random
import sqlite3

def calculate_angle(x0, y0, x1, y1):
    """Calculate the angle (in degrees) of the vector with respect to the x-axis."""
    radians = np.arctan2(y1 - y0, x1 - x0)
    return np.degrees(radians)

def accuracy_metric(predictions, labels, tolerance_degrees=5):
    """Compute custom accuracy based on angle difference within a tolerance."""
    correct_predictions = 0
    total_predictions = predictions.shape[0]
    
    for i in range(total_predictions):
        pred_angle = 180+calculate_angle(predictions[i, 0], predictions[i, 1], predictions[i, 2], predictions[i, 3])
        true_angle = calculate_angle(labels[i, 0], labels[i, 1], labels[i, 2], labels[i, 3])
        print("angle: ", pred_angle, " : ", true_angle, " : ", abs(pred_angle - true_angle))
        if abs(pred_angle - true_angle) <= tolerance_degrees:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def read_data(path_database, shuffle_it=True):
    conn = sqlite3.connect(path_database)
    
    # Create a cursor object
    cursor = conn.cursor()
    
    sql_query = """SELECT name FROM sqlite_master  
      WHERE type='table';"""
    
    cursor.execute(sql_query)
    
    # print(cursor.fetchall())
    
    data_list = []
    with conn:
        cursor.execute("SELECT * FROM pizza_table")
        # print(cursor.fetchall())
        for res in cursor.fetchall():
            data_list.append(res)
    
    if shuffle_it:
        random.shuffle(data_list)
            
    return data_list

def split_data(data_list):
    image_list = []
    label_list = []
    for data in data_list:
        _, name, x0, y0, x1, y1 = data
        image_list.append(name)
        label_list.append((x0, y0, x1, y1))
        
    # split the dataset into 80% train data and 20% test and validation data
    n = len(image_list)
    n_train = math.floor(n*0.8)
    X_train = image_list[0:n_train]  
    y_train = label_list[0:n_train]
    X_val = image_list[n_train:n]
    y_val = label_list[n_train:n]
    
    return X_train, y_train, X_val, y_val