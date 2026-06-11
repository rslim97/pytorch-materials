# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import torch
import torchvision.transforms as T

# import torchvision.transforms.functional as T

# import torch.optim as optim

import os

from PIL import Image, ImageOps

# import random

# import any other library you need below this line


# Here is a helper function to calculate the mean and standard deviation of the images
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        data = data.float()
        # Sum over all batches, height, and width; keep the channel dimension
        channels_sum += torch.sum(data, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(data**2, dim=[0, 2, 3])
        print(channels_squared_sum > channels_sum)
        num_batches += data.shape[0]  # Add the batch size

    mean = channels_sum / (
        num_batches
        * dataloader.dataset[0][0].size(1)
        * dataloader.dataset[0][0].size(2)
    )
    std = (
        channels_squared_sum
        / (
            num_batches
            * dataloader.dataset[0][0].size(1)
            * dataloader.dataset[0][0].size(2)
        )
        - mean**2
    ) ** 0.5

    return mean, std


class Cell_data(Dataset):
    def __init__(
        self, data_dir, img_size, train="True", train_test_split=0.8, augment_data=True
    ):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        # todo
        # initialize the data class
        self.data_dir = data_dir
        self.img_size = img_size
        self.train = train
        self.train_test_split = train_test_split
        self.augment_data = augment_data

        img_dir = os.path.join(self.data_dir, "cells", "scans")
        label_dir = os.path.join(self.data_dir, "cells", "labels")
        print("label_dir", label_dir)
        img_files = sorted(os.listdir(img_dir))
        label_files = sorted(os.listdir(label_dir))
        self.images = []
        self.masks = []

        for file_name in img_files:
            img_path = os.path.join(img_dir, file_name)
            img = Image.open(img_path)
            # print('img.size', img.size)  # (1024, 1024)
            self.images.append(img)
        for file_name in label_files:
            label_path = os.path.join(label_dir, file_name)
            label = Image.open(label_path)
            self.masks.append(label)

    def zoom(self, img, s=2):
        w, h = img.size
        assert w == h
        img = ImageOps.scale(img, s)
        rw, rh = img.size
        assert rw == rh
        diff = (rw - w) // s
        img = ImageOps.crop(img, diff)
        return img

    # def normalize(self, img):
    #     arr_min = np.min(img)
    #     arr_max = np.max(img)

    #     scaled = (img - arr_min) / (arr_max - arr_min)
    #     arr_new = 2 * scaled - 1
    #     return arr_new

    def __getitem__(self, idx):
        # todo
        # load image and mask from index idx of your data
        img = self.images[idx]
        mask = self.masks[idx]

        # Data augmentation part
        if self.train:
            if self.augment_data:
                augment_mode = np.random.randint(0, 4)
                if augment_mode == 0:
                    # todo
                    # flip image vertically
                    img = ImageOps.flip(img)
                    mask = ImageOps.flip(mask)
                elif augment_mode == 1:
                    # todo
                    # flip image horizontally
                    img = ImageOps.mirror(img)
                    mask = ImageOps.mirror(mask)
                elif augment_mode == 2:
                    # todo
                    # zoom image
                    scale = np.random.choice(np.arange(2, 5))
                    img = self.zoom(img, scale)
                    mask = self.zoom(mask, scale)
                else:
                    # todo
                    # rotate image
                    angle = np.random.choice(np.arange(1, 360))
                    img = img.rotate(angle)
                    mask = mask.rotate(angle)
        # resize image to desired size
        img = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))
        # todo
        # return image and mask in tensors
        # print(mask.size)
        transform = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize(0.619, 0.216),  # Normalize mean, std from helper function by 255
            ]
        )
        # img, mask = transform(img), torch.from_numpy(np.array(mask)).unsqueeze(0)
        img, mask = transform(img), torch.tensor(np.array(mask)).unsqueeze(0)
        # img, mask = transform(img), transform(mask)

        # print(img.shape)
        # print(mask.shape)
        # print(img)
        # print(mask)
        return img, mask

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    # Project path
    # proj_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, '../')))
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    print("proj_dir", proj_dir)
    data_root = os.path.join(proj_dir, "data")
    print("data_dir", data_root)

    dataset = Cell_data(data_root, 128)

    # data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)

    # mean, std = get_mean_and_std(data_loader)
    # print(mean)
    # print(std)

    fig = plt.figure(figsize=(10, 10))
    k = 4
    for i, sample in enumerate(dataset):
        img, mask = sample
        print("img.shape", img.shape)  # (1, 128, 128)
        print("mask.shape", mask.shape)  # (1, 128, 128)
        # print(np.mean(img.numpy()))
        # print(np.min(img.numpy()))
        # print(np.max(img.numpy()))
        ax1 = fig.add_subplot(k, 2, 2 * i + 1)
        ax1.imshow(img.numpy().transpose(1, 2, 0))
        ax2 = fig.add_subplot(k, 2, 2 * i + 2)
        ax2.imshow(mask.numpy().transpose(1, 2, 0))
        if i == k - 1:
            plt.tight_layout()
            plt.show()
            break
