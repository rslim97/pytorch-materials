import os
import torch.utils.data as data
import argparse
import torchvision.transforms as transforms
from utils import gt_creator
from dataset import Dataset
from dataset import CAR_CLASSES
import cv2
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proj_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        type=str,
        help="project_directory",
    )
    parser.add_argument(
        "--dataset_dir", default="./coda_small", type=str, help="dataset_directory"
    )
    args = parser.parse_args()
    train_dataset = Dataset(args, split="train", transform=[transforms.ToTensor()])
    train_loader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    num_classes = len(CAR_CLASSES)
    train_iter = iter(train_loader)
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    for i in range(3):
        img, target = next(train_iter)
        batch_size, h0, w0, _ = img.shape
        # print('img.shape', img.shape)  # n, h, w, c
        target = [label.tolist() for label in target]
        target = gt_creator(
            input_size=Dataset.image_size,
            stride=4,
            num_classes=num_classes,
            label_lists=target,
        )
        hs, ws = int(Dataset.image_size / 4), int(Dataset.image_size / 4)
        target = target.reshape(1, hs, ws, num_classes + 4 + 1)
        target_ = target[:, :, :, :num_classes].sum(axis=3).transpose(1, 2, 0)
        target_ = cv2.resize(target_, (h0, w0))
        target_ = np.repeat(target_[..., np.newaxis], repeats=3, axis=2) * 255

        # Need to undo image normalization before plot
        img = img.detach().cpu().numpy().squeeze()
        img = ((img * std) + mean).astype(np.uint8)
        img = img[:, :, (2, 1, 0)]

        img_combined = np.concatenate((img, target_), axis=1).astype(np.uint8)

        cv2.imshow("image and target", img_combined)
        cv2.waitKey(0)
        cv2.imwrite(
            os.path.join(args.proj_dir, f"image_and_target_{i}.jpg"), img_combined
        )
        cv2.destroyAllWindows()
