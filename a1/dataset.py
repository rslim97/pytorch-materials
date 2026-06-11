import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as T
from torchvision import datasets, models

import matplotlib.pyplot as plt
import time
import os

# https://www.cnblogs.com/cxq1126/p/13697361.html

#Here is a helper function to calculate the mean and standard deviation of the images
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        data = data.float()
        # Sum over all batches, height, and width; keep the channel dimension
        channels_sum += torch.sum(data, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(data ** 2, dim=[0, 2, 3])
        # print(channels_squared_sum > channels_sum)
        num_batches += data.shape[0]  # Add the batch size

    mean = channels_sum / (num_batches * dataloader.dataset[0][0].size(1) * dataloader.dataset[0][0].size(2))
    std = (channels_squared_sum / (num_batches * 
                                dataloader.dataset[0][0].size(1) * 
                                dataloader.dataset[0][0].size(2)) - mean ** 2) ** 0.5

    return mean, std

if __name__ == '__main__':
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_dir, 'dataset')
    input_size = 224
    batch_size = 64

    mean = torch.tensor([0.4767, 0.4616, 0.4160])
    std = torch.tensor([0.2284, 0.2287, 0.2540])
    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]),
        'test': T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    }
    
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                    data_transforms['train'])
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                    data_transforms['test'])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    
    # mean, std = get_mean_and_std(train_loader)
    # print('mean: {}, std: {}'.format(mean, std))

    img = next(iter(train_loader))[0]
    label = next(iter(train_loader))[1]
    print('torch.mean(img, dim=(2, 3))', torch.mean(img, dim=(2, 3),keepdim=True))
    print('torch.mean(img, dim=(2, 3))', torch.mean(img, dim=(2, 3),keepdim=True).shape)
    print('torch.min(img)', torch.min(img))
    print('torch.max(img)', torch.max(img))
    print(img.shape)
    print(label.shape)
    print('label', label)
    fig, ax = plt.subplots(1, batch_size // 2, figsize=(15, 10))
    for data in train_loader:
        x, y = data
        print('x.shape', x.shape)
        print('y.shape', y.shape)
        for j in range(batch_size // 2):
            img = x[j].numpy().transpose(1, 2, 0)
            ax[j].imshow(img)
            ax[j].axis('off')
            ax[j].set_title(f'{y[j]}')
        break
    
    plt.show()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)