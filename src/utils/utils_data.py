import torch
import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

#write a function to calculate mean and std of cifar10
def get_mean_std(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_mean_std_all(dataloader):
    mean = 0.
    var = 0.
    pixel_count = 0.0
    nb_samples = 0.
    for x_data, _ in dataloader:
        batch_samples = x_data.size(0)
        x_data = x_data.view(batch_samples, x_data.size(1), -1)
        mean += x_data.mean(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    for x_data, _ in dataloader:
        batch_samples = x_data.size(0)
        x_data = x_data.view(batch_samples, x_data.size(1), -1)
        var += ((x_data - mean.unsqueeze(1))**2).sum([0,2])
        pixel_count += x_data.nelement()
    std = torch.sqrt(var / (pixel_count-1))
    return mean, std

import numpy as np
def get_mean_std_3(dataloader):
    mean = 0.0
    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)

    std = 0.0
    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(std / (len(dataloader.dataset)*32*32))
    return mean, std


from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode
# transform_basic = Compose([ToTensor()])
# # tensor([0.4909, 0.4816, 0.4459]) tensor([0.2162, 0.2135, 0.2342])
# transform_blurred = Compose([ToTensor(), Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None)])
# dataset = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=True, download=True, transform=transform_blurred)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
# mean, std = get_mean_std_3(dataloader)
# print(mean, std)


from src.utils import common

def get_held_out_data(dataset_name, nb_samples=50):
    train_dataset, _, _ = common.DATASET_NAME_MAP[dataset_name]()
    x_data = np.array(train_dataset.data)
    y_data = np.array(train_dataset.targets)
    num_classes = len(np.unique(y_data))
    nb_samples_per_class = nb_samples // num_classes
    idxs = []
    for i in range(num_classes):
        idxs_i = np.where(y_data == i)[0]
        sampled_idxs_i = np.random.choice(idxs_i, size=nb_samples_per_class, replace=False)
        idxs.append(sampled_idxs_i)
        
    idxs = np.concatenate(idxs)
    x_data = x_data[idxs]
    y_data = y_data[idxs]
    
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save(f'data/{dataset_name}_held_out_x.npy', x_data)
    np.save(f'data/{dataset_name}_held_out_y.npy', y_data)
    return x_data, y_data