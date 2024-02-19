#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:18:35 2024

@author: becky
"""

import torch
import numpy as np

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