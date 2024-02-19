#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:41:58 2024

@author: becky
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sqlite3
import numpy as np
from PIL import Image
from model import *
from read_data import * 
from utils import *
from dataset import ToTensor, Rescale, Rotate, PizzaSliceDataset
from early_stop_pytorch import EarlyStopping
import matplotlib.pyplot as plt
 

# # Training function
def train(model, train_loader, criterion, optimizer):
    train_loss = 0.0
    model.train()
    for i, sample in enumerate(train_loader):
        images = sample['image'].double()
        labels = sample['keypoints']
        
        # Load images as tensors with gradient accumulation abilities
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)
        outputs_reshaped = outputs.view(-1, 2, 2)
        # Calculate Loss: softmax --> MSE loss
        loss = criterion(outputs_reshaped, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss



# Validation function
def validate(model, val_loader, criterion, tolerance_degrees=5):
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        # Calculate Accuracy  
        val_loss = 0.0
        
        correct = 0
        total = 0
        # Iterate through test dataset
        for sample_val in val_loader:
            
            images = sample_val['image'].double()
            labels = sample_val['keypoints']
            labels = labels.view(-1, 4)
            labels_np = labels.detach().numpy()
            # Load images to tensors with gradient accumulation abilities
            images = images.requires_grad_()
            
            # # Forward pass only to get logits/output
            output = model(images)
            
            predictions = output.detach().numpy()
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            accuracy = accuracy_metric(predictions, labels_np, tolerance_degrees=tolerance_degrees)
            
    return val_loss, accuracy

              

if __name__ == "__main__":
    plt.close("all")
    path_database = "data/pizza_database.db"
    
    data_list = read_data(path_database, False)

    X_train, y_train, X_val, y_val = split_data(data_list)
    
    path_dir = "data"
    
    transform = transforms.Compose([Rescale((150, 150)), Rotate(45), ToTensor()]) #, Rotate(45), ToTensor()])
    
    train_dataset = PizzaSliceDataset(path_dir, X_train, y_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    dataset_val = PizzaSliceDataset(path_dir, X_val, y_val, transform=transform)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True)
    
    # Loss function
    criterion = nn.MSELoss()   

    model = PizzaSliceNet() 
    model.double()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 20
    n_iters = 30
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    
    # Initialize the EarlyStopping object
    early_stopping = EarlyStopping(patience=50, verbose=True, delta=0.001)
    loss_train_list = []
    loss_val_list = []
    accuracy = []
    for epoch in range(num_epochs):
            
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, acc = validate(model, val_loader, criterion, tolerance_degrees=100)
        
        loss_train_list.append(train_loss)
        loss_val_list.append(val_loss)
        accuracy.append(acc)
        # Calculate average losses
        train_loss_ave = train_loss / len(train_loader)
        val_loss_ave = val_loss / len(val_loader)
        
        # Print training/validation statistics 
        print(f'Epoch {epoch+1}: \tTraining Loss: {train_loss_ave:.6f} \tValidation Loss: {val_loss_ave:.6f}')
    
        # Early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))

plt.figure()
plt.plot(range(0,len(loss_train_list)), loss_train_list) 
plt.plot(range(0,len(loss_val_list)), loss_val_list, 'r')                 
    
    
    
