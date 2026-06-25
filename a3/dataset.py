import os
import json
import argparse

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2

# CAR_CLASSES = ["car", "pedestrian", "cyclist", "truck", "tram"]
# CAR_CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram']
CAR_CLASSES = ["pedestrian", "cyclist", "car", "truck", "tram"]


class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, args, split, transform):
        print("Dataset Initialization")
        self.args = args
        proj_dir = args.proj_dir
        data_dir = args.dataset_dir
        self.images_root = os.path.join(proj_dir, data_dir, split, "images")
        print("self.images_root", self.images_root)
        if split == "train":
            self.train = False
        else:
            self.train = False

        self.transform = transform
        self.f_names, self.boxes, self.labels = [], [], []
        self.mean = [123.675, 116.280, 103.530]  # RGB
        self.std = [58.395, 57.120, 57.375]
        annotations_path = os.path.join(
            proj_dir, data_dir, "annotations", "instance_" + split + ".json"
        )
        print("annotations_path", annotations_path)
        annotations = load_json(annotations_path)

        for annotation in annotations["annotations"]:
            bboxes = []
            labels = []
            remove = []
            for i, bbox in enumerate(annotation["bbox"]):
                x1, y1, x2, y2 = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[0] + bbox[2]),
                    float(bbox[1] + bbox[3]),
                )  # xmin, ymin, xmax, ymax
                if (
                    x1 >= 0
                    and y1 >= 0
                    and x2 >= 0
                    and y2 >= 0
                    and x2 > x1
                    and y2 > y1
                    and x2 < 1281
                    and y2 < 720
                ):
                    bboxes.append([x1, y1, x2, y2])
                else:
                    remove.append(i)
            for i, cat_id in enumerate(annotation["category_id"]):
                if i not in remove:
                    labels.append(cat_id)
            if len(bboxes) > 0:  # Only save include with bounding box to train data
                self.f_names.append(annotation["image_name"])
                self.boxes.append(torch.tensor(bboxes))  # list<torch.tensor>
                self.labels.append(
                    torch.tensor(labels, dtype=torch.int32)
                )  # list<torch.tensor>

        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        f_name = self.f_names[idx]
        img = cv2.imread(os.path.join(self.images_root, f_name))
        boxes = self.boxes[idx].clone()  # clone for torch tensors
        labels = self.labels[idx].clone()  # clone for torch tensors
        # print('labels.shape', labels.shape)
        assert img is not None

        if self.train:
            # img = self.random_bright(img)
            img, boxes = random_flip(img, boxes)
            img, boxes = randomScale(img, boxes)
            img = randomBlur(img)
            img, boxes, labels = randomShift(img, boxes, labels)
            img, boxes, labels = randomCrop(img, boxes, labels)

        h, w, c = img.shape
        # print(boxes.shape)
        # print(labels.shape)
        boxes /= torch.tensor([w, h, w, h]).expand_as(boxes)
        # Convert image to rgb from bgr
        img = img[:, :, (2, 1, 0)]
        img = subMeanDividedStd(img, self.mean, self.std)
        img = cv2.resize(img, (self.image_size, self.image_size))
        # print('boxes.shape', boxes.shape)
        # print('labels.unsqueeze(1)', labels.unsqueeze(1).shape)
        target = torch.hstack((boxes, labels.unsqueeze(1)))
        # print('target', target)
        # return torch.from_numpy(img).unsqueeze(0), target.unsqueeze(0)  # (1, 3, H, W), (1, num_boxes, 5)
        return torch.from_numpy(img), target
        # return torch.from_numpy(img).permute(2, 0, 1), target  # (1, 3, H, W), (1, num_boxes, 5)

    def __len__(self):
        return self.num_samples


def randomBlur(bgr):
    if random.random() < 0.5:
        bgr = cv2.blur(bgr, (5, 5))
    return bgr


def randomShift(bgr, boxes, labels):
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    if random.random() < 0.5:
        height, width, c = bgr.shape
        after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)

        if shift_x >= 0 and shift_y >= 0:
            after_shfit_image[int(shift_y) :, int(shift_x) :, :] = bgr[
                : height - int(shift_y), : width - int(shift_x), :
            ]
        elif shift_x >= 0 and shift_y < 0:
            after_shfit_image[: height + int(shift_y), int(shift_x) :, :] = bgr[
                -int(shift_y) :, : width - int(shift_x), :
            ]
        elif shift_x < 0 and shift_y >= 0:
            after_shfit_image[int(shift_y) :, : width + int(shift_x), :] = bgr[
                : height - int(shift_y), -int(shift_x) :, :
            ]
        elif shift_x < 0 and shift_y < 0:
            after_shfit_image[: height + int(shift_y), : width + int(shift_x), :] = bgr[
                -int(shift_y) :, -int(shift_x) :, :
            ]

        shift_xy = torch.tensor(
            [[int(shift_x), int(shift_y)]], dtype=torch.float32
        ).expand_as(center)
        center = center + shift_xy
        mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.tensor(
            [[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]],
            dtype=torch.float32,
        ).expand_as(boxes_in)
        boxes_in = boxes_in + box_shift
        labels_in = labels[mask.view(-1)]
        return after_shfit_image, boxes_in, labels_in
    return bgr, boxes, labels


def randomScale(bgr, boxes):
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        height, width, c = bgr.shape
        bgr = cv2.resize(bgr, (int(width * scale), height))
        scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        return bgr, boxes
    return bgr, boxes


def randomCrop(bgr, boxes, labels):
    if random.random() < 0.5:
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        height, width, c = bgr.shape
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(0, width - w)
        y = random.uniform(0, height - h)
        x, y, h, w = int(x), int(y), int(h), int(w)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)

        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
        boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
        boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
        boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

        labels_in = labels[mask.view(-1)]
        img_croped = bgr[y : y + h, x : x + w, :]
        return img_croped, boxes_in, labels_in
    return bgr, boxes, labels


def subMean(bgr, mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr


def subMeanDividedStd(rgb, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    rgb = (rgb - mean) / std
    return rgb


def random_flip(im, boxes):
    if random.random() < 0.5:
        im_lr = np.fliplr(im).copy()
        h, w, _ = im.shape
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
        return im_lr, boxes
    return im, boxes


def random_bright(im, delta=16):
    alpha = random.random()
    if alpha > 0.3:
        im = im * alpha + random.randrange(-delta, delta)
        im = im.clip(min=0, max=255).astype(np.uint8)
    return im


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proj_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        type=str,
        help="project_directory",
    )
    parser.add_argument(
        "--dataset_dir", default="coda_small", type=str, help="dataset_directory"
    )
    args = parser.parse_args()
    train_dataset = Dataset(args, split="train", transform=[transforms.ToTensor()])

    train_loader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    train_iter = iter(train_loader)
    for i in range(5):
        img, target = next(train_iter)
        print("img.shape", img.shape)
        print("target.shape", target.shape)
        # print(img, target)
        # target = [label.tolist() for label in target]

        # target = gt_creator(Dataset.image_size, 4, len(CAR_CLASSES), target)
        # print(target)


if __name__ == "__main__":
    main()  # For debug
