import os
import torch
import torch.nn as nn

# import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# import torchvision.transforms as T
# import torchvision.transforms.functional as T

import torch.optim as optim

from model.unet import UNet
from model.unet_pretrained import Unet_pretrained_encoder
from dataset import Cell_data
from loss import FocalLoss

from copy import deepcopy
import random

torch.manual_seed(34)
random.seed(34)
np.random.seed(34)


def train_and_val(model, config):
    valid_loss_min = np.Inf
    for e in range(config["epochs"]):
        epoch_loss = 0
        model.train()
        for i, data in enumerate(trainloader):
            image, label = data
            # print('image.shape', image.shape)  # (batch_size, 1, 512, 512)
            # print('label.shape', label.shape)  # (batch_size, 1, 512, 512)
            # print('torch.unique(label)', torch.unique(label))
            # image = image.to(device)
            image = image.float().to(device)
            # print(torch.min(image))
            # print(torch.max(image))
            # print(torch.mean(image))
            label = label.squeeze(1).long().to(device)
            # print('label.shape', label.shape)
            # print('image.shape', image.shape)
            # print(type(image))
            # print(image.dtype)
            pred = model(image)

            crop_y = (label.shape[1] - pred.shape[2]) // 2
            crop_x = (label.shape[2] - pred.shape[3]) // 2

            label = label[
                :, crop_x : label.shape[2] - crop_x, crop_y : label.shape[1] - crop_y
            ]

            # print('pred.shape', pred.shape)
            # print('label.shape', label.shape)
            # print(torch.min(label))
            # print(torch.max(label))
            loss = criterion(pred, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            print("batch %d --- Loss: %.4f" % (i, loss.item() / config["batch_size"]))

            # Release memory
            del image, label, pred
            torch.cuda.empty_cache()

        print(
            "Epoch %d / %d --- Loss: %.4f"
            % (e + 1, config["epochs"], epoch_loss / len(trainset))
        )
        # metrics = {
        #     'train_loss': epoch_loss
        # }
        # wandb.log(metrics)

        model.eval()

        total = 0
        valid_acc = 0
        valid_loss = 0

        with torch.no_grad():
            for i, data in enumerate(testloader):
                image, label = data

                image = image.float().to(device)
                # print(label.shape)
                label = label.squeeze(1).long().to(device)

                pred = model(image)
                crop_x = (label.shape[1] - pred.shape[2]) // 2
                crop_y = (label.shape[2] - pred.shape[3]) // 2

                label = label[
                    :,
                    crop_x : label.shape[2] - crop_x,
                    crop_y : label.shape[1] - crop_y,
                ]
                # print(label[:, :5, :5])
                loss = criterion(pred, label)
                valid_loss += loss.item()

                _, pred_labels = torch.max(pred, dim=1)

                total += label.shape[0] * label.shape[1] * label.shape[2]
                valid_acc += (pred_labels == label).sum().item()

                print(
                    "Accuracy: %.4f ---- Loss: %.4f"
                    % (valid_acc / total, valid_loss / len(testset))
                )

                del image, label, pred
                torch.cuda.empty_cache()

            metrics = {
                "val_loss": valid_loss / len(testset),
                "val_acc": valid_acc / total,
                "lr": config["lr"],
            }
            wandb.log(metrics)

            if valid_loss <= valid_loss_min:
                print(
                    "Validation loss decreased ({:.5f}->{:.5f}). Saving model ... ".format(
                        valid_loss_min, valid_loss
                    )
                )

                torch.save(
                    {
                        "model_state_dict": deepcopy(model.state_dict()),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_fname,
                )


if __name__ == "__main__":
    import wandb

    # os.environ['WANDB_API_KEY'] = ''
    wandb.login(
        key="",
        relogin=True,
    )

    run_config = {
        "model": "UNet",
        "epochs": 90,
        "batch_size": 4,
        "lr": 1e-3,
        "dropout": random.uniform(0.01, 0.80),
        "image_size": 512,
        "gpu": True,
        "weight": torch.tensor([0.3, 0.7]),
    }

    # run_config = {
    #     "model": "UNet-pretrained",
    #     "epochs": 90,
    #     "batch_size": 4,
    #     "lr": 1e-4,
    #     "dropout": random.uniform(0.01, 0.80),
    #     "image_size": 512,
    #     "gpu": True,
    #     "weight": torch.tensor([0.3, 0.7]),
    # }

    run = wandb.init(
        project="Cell-Segmentation",
        name=run_config["model"],
        config=run_config,
    )
    # Parameters

    # # learning rate
    # lr = 1e-4

    # # number of training epochs
    # epoch_n = 30

    # # input image-mask size
    # image_size = 512

    # # training batch size
    # batch_size = 4

    # # use checkpoint model for training
    # load = False

    # # use GPU for training
    # gpu = True

    # root directory of project
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_dir, "data")
    ckpt_dir = os.path.join(proj_dir, "checkpoints")
    results_dir = os.path.join(proj_dir, "results")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    trainset = Cell_data(
        data_dir=data_dir, img_size=run_config["image_size"], train=True
    )
    trainloader = DataLoader(
        trainset, batch_size=run_config["batch_size"], shuffle=True
    )

    testset = Cell_data(
        data_dir=data_dir, img_size=run_config["image_size"], train=False
    )
    testloader = DataLoader(testset, batch_size=run_config["batch_size"])

    device = torch.device("cuda:0" if run_config["gpu"] else "cpu")
    print("device", device)

    model = UNet(n_classes=2).to(device)
    # model = Unet_pretrained_encoder(n_classes=2).to(device)
    # print(model)

    # # Freeze layers up through layer 1
    # for name, layer in model.named_children():
    #     print('name ', name)
    #     if name in ['input_block']:
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #         for m in layer.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eval()

    # # Print layer freezing status
    # for name, param in model.named_parameters():
    #     print(f"{name} requires_grad={param.requires_grad}")
    fname = "checkpoint.pt"
    ckpt_fname = os.path.join(ckpt_dir, fname)

    if torch.cuda.is_available():
        model.cuda()

    # if load:
    #     print("loading model")
    #     model.load_state_dict(torch.load("checkpoint.pt"), map_location=device)

    # print(model)
    # criterion = nn.CrossEntropyLoss(weight=run_config['weight'].to(device))
    criterion = FocalLoss(weight=run_config["weight"].to(device), reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=run_config["lr"], weight_decay=0.0005)

    train_and_val(model, run_config)

    model_arch = str(model)
    wandb.save(os.path.join(results_dir, "model_arch.txt"))
    run.finish()

    # Testing and visualization
    model = UNet(n_classes=2).to(device)
    # model = Unet_pretrained_encoder(n_classes=2).to(device)

    checkpoint = torch.load(ckpt_fname, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_images = []
    output_masks = []
    output_labels = []

    with torch.no_grad():
        for i in range(len(testset)):
            image, labels = testset.__getitem__(i)

            # print('image.shape', image.shape)
            # print('labels.shape', labels.shape)

            input_image = image.float().unsqueeze(0).to(device)
            pred = model(input_image)

            # print('pred.shape', pred.shape)
            # print('torch.max(pred, dim = 1)[1]', torch.max(pred, dim = 1))
            output_mask = (
                torch.max(pred, dim=1)[1].cpu().squeeze(0).detach().cpu().numpy()
            )

            # print('output_mask.shape', output_mask.shape)
            # print('output_mask', output_mask)

            crop_y = (labels.shape[1] - output_mask.shape[0]) // 2
            crop_x = (labels.shape[2] - output_mask.shape[1]) // 2
            labels = (
                labels[
                    :,
                    crop_x : labels.shape[1] - crop_x,
                    crop_y : labels.shape[2] - crop_y,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            # print('labels.shape', labels.shape)
            # print('output_mask.shape', output_mask.shape)

            test_images.append(image.detach().cpu().numpy())
            output_masks.append(output_mask)
            output_labels.append(labels)

    fig, subs = plt.subplots(len(testset) // 4, 3, figsize=(30, 30))

    # print(output_masks[0].shape)
    # print(output_masks[0])
    # print(np.max(output_masks[0]))
    # print(np.min(output_masks[0]))

    for idx, sub in zip(np.arange(len(testset) // 4), subs.flatten()):

        print(np.max(output_masks[idx]))
        print(np.min(output_masks[idx]))
        subs[idx, 0].imshow(output_labels[idx].transpose(1, 2, 0))
        subs[idx, 0].axis("off")
        subs[idx, 1].imshow(test_images[idx].transpose(1, 2, 0))
        subs[idx, 1].axis("off")
        subs[idx, 2].imshow(output_masks[idx])
        subs[idx, 2].axis("off")
    # for i in range(testset.__len__()):
    #     axes[i, 0].imshow(output_labels[i].transpose(1, 2, 0))
    #     axes[i, 0].axis('off')
    #     axes[i, 1].imshow(output_masks[i])
    #     axes[i, 1].axis('off')

    plt.savefig(os.path.join(results_dir, "Cell-Segmentation-test.png"))
    plt.show()
