import os
import argparse

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from dataset import Dataset, CAR_CLASSES

from model.ctdet import Model
from utils import get_loss, gt_creator
from tqdm import tqdm


# def test():
#     x = torch.rand(5, 3, 448, 448)
#     model = Model(4, 10)
#     out = model(x)


# if __name__ == '__main__':
#     test()


if __name__ == "__main__":
    device = torch.device("cuda") if "gpu" else torch.device("cpu")
    num_epochs = 3

    model = Model(num_classes=len(CAR_CLASSES), topk=10).to(device)

    img = torch.rand(1, 448, 448, 3).permute(0, 3, 1, 2).to(device)
    cls_pred, txty_pred, twth_pred = model(img)
    print(cls_pred.shape)
    print(txty_pred.shape)
    print(twth_pred.shape)

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
    parser.add_argument(
        "--ckpt_dir", default="checkpoints", type=str, help="checkpoints directory"
    )
    args = parser.parse_args()
    train_dataset = Dataset(args, split="train", transform=[transforms.ToTensor()])
    val_dataset = Dataset(args, split="val", transform=[transforms.ToTensor()])
    test_dataset = Dataset(args, split="test", transform=[transforms.ToTensor()])
    proj_dir = args.proj_dir
    ckpt_dir = os.path.join(proj_dir, args.ckpt_dir)
    fname = "ctdet"
    ckpt_fname = os.path.join(ckpt_dir, fname + ".pt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    train_loader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    val_loss_min = np.float64("inf")
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for i, data in tqdm(enumerate(train_loader)):
            image, target = data
            batch_size = image.size(0)
            optimizer.zero_grad()
            image = image.permute(0, 3, 1, 2).to(device)
            target = [label.tolist() for label in target]

            target = gt_creator(
                448, stride=4, num_classes=len(CAR_CLASSES), label_lists=target
            )
            target = torch.tensor(target).float().to(device)
            pred_cls, pred_txty, pred_twth = model(image)
            total_loss = get_loss(
                pred_cls,
                pred_txty,
                pred_twth,
                label=target,
                num_classes=len(CAR_CLASSES),
            )
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * batch_size
            # print(total_loss.data)
            del image, target, pred_cls, pred_txty, pred_twth
            torch.cuda.empty_cache()

        train_loss /= len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            for i, data in tqdm(enumerate(val_loader)):
                image, target = data
                batch_size = image.size(0)
                optimizer.zero_grad()
                image = image.permute(0, 3, 1, 2).to(device)
                target = [label.tolist() for label in target]

                target = gt_creator(
                    448, stride=4, num_classes=len(CAR_CLASSES), label_lists=target
                )
                target = torch.tensor(target).float().to(device)
                pred_cls, pred_txty, pred_twth = model(image)
                total_loss = get_loss(
                    pred_cls,
                    pred_txty,
                    pred_twth,
                    label=target,
                    num_classes=len(CAR_CLASSES),
                )
                val_loss += total_loss.item() * batch_size
                del image, target, pred_cls, pred_txty, pred_twth
                torch.cuda.empty_cache()

            if val_loss <= val_loss_min:
                print(
                    "Validation loss decreased ({:.5f}->{:.5f}). Saving model".format(
                        val_loss_min, val_loss
                    )
                )
                torch.save(model.state_dict(), ckpt_fname)
                val_loss_min = val_loss

    test_model = Model(num_classes=len(CAR_CLASSES), topk=10).to(device)
    model_state_dict = torch.load(ckpt_fname, weights_only=True)
    if model_state_dict:
        test_model.load_state_dict(model_state_dict)
        print("Loaded model...")
