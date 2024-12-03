import struct

from abc import ABC, abstractmethod

import numpy as np
from array import array

import matplotlib.pyplot as plt


class MnistDataset(ABC):
    def __init__(
        self, 
        img_path, 
        labels_path,
    ):
        self.CLASSES = 10
        self.img_path = img_path
        self.labels_path = labels_path
        self.load_data()

    def read_images_labels(self, images_path, labels_path):        
        labels = []
        with open(labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img   
            
        labels = self.convert_to_indeces(labels)       
        
        return images, labels
            
    def load_data(self):
        self.x_data, self.y_data = self.read_images_labels(self.img_path, self.labels_path)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return np.array(self.x_data[idx]).flatten(), self.y_data[idx]
    
    def convert_to_indeces(self, gt):
        ground_truths = []
        for idxs in gt:
            indeces = np.zeros(self.CLASSES, dtype=np.float32).tolist()
            indeces[idxs] = 1.
            ground_truths.append(indeces)

        return np.array(ground_truths)
    
    def convert_to_label(self, indeces):
        label = np.argmax(indeces).item()
        return label
    
    def show_images(self, images, title_texts):
        cols = 5
        rows = int(len(images) / cols) + 1
        plt.figure(figsize=(30, 20))
        index = 1    
        plt.subplot(rows, cols, index)        
        plt.imshow(images, cmap=plt.cm.gray)
        title_texts = self.convert_to_label(title_texts)
        if (title_texts != ''):
            plt.title(title_texts, fontsize = 15);  


             