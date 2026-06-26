import os
import argparse

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from dataset import Dataset, CAR_CLASSES

from model.ctdet import Model
from utils import get_loss, gt_creator, detection_collate
from tqdm import tqdm

import cv2

# def test():
#     x = torch.rand(5, 3, 448, 448)
#     model = Model(4, 10)
#     out = model(x)


# if __name__ == '__main__':
#     test()

if __name__ == "__main__":
    device = torch.device("cuda") if "gpu" else torch.device("cpu")
    num_epochs = 150

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
    results_dir = os.path.join(proj_dir, "results")
    fname = "ctdet"
    ckpt_fname = os.path.join(ckpt_dir, fname + ".pt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=1,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=detection_collate,
        num_workers=1,
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    print("len(train_loader)", len(train_loader))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
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
            # print('target', target)
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
                    "Epoch {:03d} Validation loss decreased ({:.5f}->{:.5f}). Saving model".format(
                        epoch + 1, val_loss_min, val_loss
                    )
                )
                torch.save(model.state_dict(), ckpt_fname)
                val_loss_min = val_loss

    test_model = Model(num_classes=len(CAR_CLASSES), topk=10).to(device)
    model_state_dict = torch.load(ckpt_fname, weights_only=True)
    if model_state_dict:
        test_model.load_state_dict(model_state_dict)
        print("Loaded model...")

    # test(test_model)
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])
    test_model.eval()
    cnt = 0
    for i, data in enumerate(test_loader):
        image, target = data
        # print('type(image)', type(image))
        batch_size, h0, w0, _ = image.shape
        image = image.permute(0, 3, 1, 2).to(device)
        # image = image.to(device)
        print("image.shape", image.shape)
        bbox_pred, score, cls_ind = test_model.predict(image)
        bbox_pred = bbox_pred * np.array([[w0, h0, w0, h0]])  # xyxy
        target_bbox = target.squeeze(0)[:, :4].detach().cpu().numpy()
        target_cls = target.squeeze(0)[:, 4].detach().cpu().numpy()
        print("bbox_target.shape", target_bbox.shape)
        target_bbox = target_bbox * np.array([[w0, h0, w0, h0]])  # xyxy
        # print('bbox_pred.shape', bbox_pred.shape)
        # image = (image.permute(0, 2, 3, 1)[0][:,:,(2,1,0)].detach().cpu().numpy()*255).astype(np.uint8)
        image = image[0].detach().permute(1, 2, 0).cpu().numpy()
        image = ((image * std) + mean).astype(np.uint8)
        # image = image[:, :, (2, 1, 0)]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(image)

        for j, box in enumerate(bbox_pred):
            if score[j] > 0.3:
                # print('type(image)', type(image))
                cls_indx = cls_ind[j]
                # print('type(cls_indx)', type(cls_indx))
                # print(type(box))
                # print(box)
                xmin, ymin, xmax, ymax = box
                # print('image.shape', image.shape)
                # print('type(image)', type(image))
                # image = image.detach()
                cv2.rectangle(
                    image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2
                )
                cv2.putText(
                    image,
                    str(CAR_CLASSES[cls_indx - 1]),
                    (int(xmin), int(ymin)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            print("score", score)
            # print(j, box)
        for j, box_target in enumerate(target_bbox):
            # print(type(target_cls[j]))
            # tgt = target
            tgt_xmin, tgt_ymin, tgt_xmax, tgt_ymax = box_target
            cv2.rectangle(
                image,
                (int(tgt_xmin), int(tgt_ymin)),
                (int(tgt_xmax), int(tgt_ymax)),
                (255, 0, 0),
                1,
            )
            cv2.putText(
                image,
                str(CAR_CLASSES[int(target_cls[j] - 1)]),
                (int(tgt_xmin), int(tgt_ymin)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(results_dir, "test" + str(cnt) + ".jpg"), image)
        cnt += 1
        if cnt >= 15:
            break
