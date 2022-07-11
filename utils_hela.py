
from __future__ import division, print_function
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from torchvision import datasets
import random
from torchvisionnewtransforms import RandomPerspective,ColorJitter,RandomAffine,RandomRotation,RandomHorizontalFlip,RandomVerticalFlip
from scipy.optimize import linear_sum_assignment as linear_assignment
#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)    
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def load_hela(data_path='./Hela_jpg/train'):
    
    train_dataset = datasets.ImageFolder(data_path, transforms.Compose([ #Transform only when get_item())
            transforms.Grayscale(num_output_channels=1), #When using ImageFolder class and with no custom loader, pytorch uses PIL to load image and converts it to RGB
            transforms.Resize(224),
            transforms.ToTensor(),

        ])) 
    
    
    imgs_tuples = train_dataset.imgs
    
    data_tensor = None
    target_tensor = []
    for i in range  (862): 
        index = i
        path,target = imgs_tuples[index]
        image = train_dataset.loader(path)
        image = transforms.Grayscale(num_output_channels=1)(image)
        image = transforms.Resize(224)(image)
        image = image.convert('RGB') 

        image = transforms.ToTensor()(image)
        if data_tensor is None:
            img = torch.unsqueeze(image, 0)   #For color image of N channels
            data_tensor = img
            target_tensor.append(target)
        else:
            img = torch.unsqueeze(image, 0)      #For color image of N channels      
            data_tensor = torch.cat((data_tensor, img),0)
            target_tensor.append(target)
        #print("image = {} target= {}".format(i,target))
        

    data_train = np.array(data_tensor)
    labels_train =   np.array(target_tensor)
    data_train = data_train.astype(np.float32)
    labels_train = labels_train.astype(np.int64)
    

    x = data_train
    y = labels_train
    print( 'Hela samples', x.shape)
    return x, y

class HelaDataset(Dataset):

    def __init__(self):

        x_tr, y_tr= load_hela()
        self.x_tr = torch.from_numpy(x_tr)
        print('x min values',torch.min(self.x_tr))
        print('x max values',torch.max(self.x_tr))
        
        self.y_tr = y_tr
       
    def __len__(self):
        return self.x_tr.shape[0]

    def __getitem__(self, idx):
        x_tr = self.x_tr[idx]
        
        x_tr = transforms.ToPILImage()(x_tr)
        x_tr = RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)(x_tr)
        x_tr = RandomHorizontalFlip()(x_tr)
        x_tr = RandomVerticalFlip(p=0.5)(x_tr)
        x_tr = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(x_tr)
        x_tr = RandomAffine(90, translate=(0.3,0.3), scale=(0.5,1.5), shear=60, resample=False, fillcolor=0)(x_tr)
        x_tr = RandomRotation(90, resample=False, expand=False, center=None)(x_tr)
        x_tr = transforms.ToTensor()(x_tr)
        
        y_tr = torch.from_numpy(np.array(self.y_tr[idx]))
        idx = torch.from_numpy(np.array(idx))
        return x_tr,y_tr,idx
    

         