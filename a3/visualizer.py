import os
import torch.utils.data as data
import argparse
import torchvision.transforms as transforms
from utils import gt_creator
from dataset import Dataset
from dataset import CAR_CLASSES
import cv2


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
    for i in range(5):
        img, target = next(train_iter)
        target = [label.tolist() for label in target]
        target = gt_creator(
            input_size=Dataset.image_size,
            stride=4,
            num_classes=num_classes,
            label_lists=target,
        )
        print("target", target)
        print("target.shape", target.shape)
        hs, ws = int(Dataset.image_size / 4), int(Dataset.image_size / 4)
        target = target.reshape(1, hs, ws, num_classes + 4 + 1)
        target_ = target[:, :, :, :num_classes].sum(axis=3).squeeze()
        print("target_.shape", target_.shape)
        # print('target.shape', target.shape)
        cv2.imshow("window", target_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(img.shape)
        # Need to undo image normalization before plot
        img = img.detach().cpu().numpy().squeeze()[:, :, (2, 1, 0)]
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\n")
