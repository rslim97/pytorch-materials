from pathlib import Path

VOC_PATH = "data"

# from path import Path
import random
import time
import csv

import numpy as np
import torch


def plot(results_fname):
    plt.figure(figsize=(10, 5))
    # Load saved data
    x = torch.load(results_fname, weights_only=True)
    train_counter, train_losses = x["train_loss"]
    valid_counter, valid_losses = x["valid_loss"]
    valid_counter, valid_maps = x["valid_mAP"]
    # val_accs = x["val_accs"]
    plt.plot(
        train_counter,
        train_losses,
        zorder=+100,
        color="#1f77b4",
        label="Train loss",
    )
    plt.plot(
        valid_counter,
        valid_losses,
        zorder=+100,
        color="#ff7f0e",
        label="Val loss",
    )
    plt.scatter(
        valid_counter,
        valid_maps,
        zorder=+200,
        color="black",
        label="Val mAP",
    )
    for i, txt in enumerate(valid_maps):
        plt.annotate(round(txt, 2), (valid_counter[i], txt+0.1), rotation=90)
    # plt.scatter(
    #     [valid_counter[np.argmin(valid_losses)]],
    #     [min(valid_losses)],
    #     color="black",
    #     zorder=+200,
    #     label=(f"Best Accuracy: {max(val_accs)*100:.2f}%"),
    # )
    plt.title(results_fname)
    plt.xlabel("Number of Examples Seen by the model")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.savefig(str(results_fname) + ".png")
    plt.show()


def seed_everything(seed: int = 444) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    SEED = 234

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"device: {device}")
    num_queries = 12
    num_encoder_layers = 1
    num_decoder_layers = 3

    d_model = 256
    nhead = 8
    dim_feedforward = 2048
    dropout = 0.1

    # backbone = "facebook/dinov2-with-registers-small"
    backbone = "facebook/dinov2-small"

    backbone_learning_rate = 1e-5
    head_learning_rate = 1e-4
    num_epochs = 50
    batch_size = 4

    import cv2
    import matplotlib.pyplot as plt
    import torch.utils.data as data

    from src.constants import COLORS, VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD
    from src.dataset import VOCDetectionDataset, collate_fn_detr

    ANNOTATION_DIR = Path("annotations")
    VOC_PATH = Path(VOC_PATH)

    VOC_ROOT = VOC_PATH / "VOCdevkit_2007"
    file_root_train = VOC_ROOT / "VOC2007" / "JPEGImages"
    annotation_file_train = ANNOTATION_DIR / "voc2007.txt"
    file_root_test = VOC_ROOT / "VOC2007test" / "JPEGImages"
    annotation_file_test = ANNOTATION_DIR / "voc2007test.txt"

    IMG_MAX_SIZE = 384
    train_dataset = VOCDetectionDataset(
        root_img_dir=file_root_train,
        dataset_file=annotation_file_train,
        train=True,
        detector_type="detr",
        backbone=backbone,
        image_size=IMG_MAX_SIZE,
        augmentation=True,
    )
    test_dataset = VOCDetectionDataset(
        root_img_dir=file_root_test,
        dataset_file=annotation_file_test,
        train=False,
        detector_type="detr",
        backbone=backbone,
        image_size=IMG_MAX_SIZE,
        augmentation=False,
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        collate_fn=collate_fn_detr,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        collate_fn=collate_fn_detr,
    )

    print(f"Loaded {len(train_dataset)} training images")
    print(f"Loaded {len(test_dataset)} test images")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Test batches per epoch: {len(test_loader)}")

    for i, data in enumerate(train_loader):
        imgs, targets = data
        print(type(imgs))
        print(type(targets))
        print(len(imgs))
        print(len(targets))
        for j in range(len(imgs)):
            # print(img[j].shape)
            img = imgs[j]  # (c, h, w)
            print(img.shape)
            target = targets[j]
            print(type(img))
            print(target)
            img = img.permute(1, 2, 0)
            img = img[..., (2, 1, 0)]
            img = img.detach().cpu().numpy()
            h0, w0, _ = img.shape
            img = (img * IMAGENET_STD) + IMAGENET_MEAN
            bboxes = target["boxes"].detach().cpu().detach()
            for k, box in enumerate(bboxes):
                box = box * np.array([w0, h0, w0, h0])
                cx, cy, w, h = box  # detr predicts cxcywh
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )
            # print(img)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("\n")
        break

    from src.detr import SimpleDETR

    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    load_network_path = None

    model = SimpleDETR(
        num_classes=len(VOC_CLASSES),
        backbone=backbone,
        num_queries=num_queries,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    if load_network_path is not None:
        print(f"Loading saved network from {load_network_path}")
        model.load_state_dict(torch.load(load_network_path, map_location=device))
    else:
        print(f"Initialized SimpleDETR with backbone: {backbone}")

    print(f"Parameter count: {count_parameters(model) / 1e6:.2f}M")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    import src.detr_loss as detr_loss_module
    from importlib import reload
    from src.detr_loss import compute_total_loss

    reload(detr_loss_module)
    matcher = detr_loss_module.HungarianMatcher(
        cost_class=1.0, cost_bbox=5.0, cost_giou=2.0
    )
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}

    # Down-weight the no-object class in the provided CE loss.
    eos_coef = 0.2
    criterion = detr_loss_module.DETRSetCriterion(
        num_classes=len(VOC_CLASSES),
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
    ).to(device)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f"Backbone learning rate: {backbone_learning_rate:.4g}")
    print(f"Head learning rate:     {head_learning_rate:.4g}")

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": head_learning_rate},
            {"params": backbone_params, "lr": backbone_learning_rate},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=0.0,
    )

    # Backbone-to-head LR ratio, used to keep backbone LR proportional during warmup/freeze
    lr_ratio = backbone_learning_rate / head_learning_rate

    eval_kwargs = {
        "test_dataset_file": str(annotation_file_test),
        "img_root": str(file_root_test),
        "iou_threshold": 0.5,
        "use_07_metric": True,
    }

    # Toggle this to True to print AP for each class.
    print_per_class_mAP = False
    # Change this if you want to save checkpoints more or less often during training.
    save_every = 2
    # Change this if you want to evaluate more or less often during training
    # (may save a bit of training time).
    eval_every = 1

    from collections import defaultdict
    from tqdm import tqdm

    from src.eval import evaluate_detr

    seed_everything(SEED)

    @torch.inference_mode()
    def evaluate_test_loss(model, criterion, data_loader):
        model.eval()
        criterion.eval()
        loss_sums = defaultdict(float)
        num_batches = 0

        for images, targets in data_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = compute_total_loss(loss_dict, weight_dict)
            loss_sums["total_loss"] += float(total_loss.item())
            for key, value in loss_dict.items():
                loss_sums[key] += float(value.item())
            num_batches += 1

        return {key: value / max(num_batches, 1) for key, value in loss_sums.items()}

    final_map = 0.0
    final_test_loss = 0.0

    global_step = 0

    run_dir = Path("runs/detr")
    run_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/detr")
    results_dir.mkdir(parents=True, exist_ok=True)

    warmup_epochs = 0.5
    steps_per_epoch = len(train_loader)
    warmup_steps = max(0, int(round(warmup_epochs * steps_per_epoch)))

    start_time = time.time()
    epoch_pbar = tqdm(range(num_epochs), desc="Training")

    freeze_backbone_epochs = 5.0

    counter = [i * len(train_loader.dataset) for i in range(num_epochs)]

    train_losses = []
    valid_losses = []
    valid_maps = []

    for epoch in epoch_pbar:
        model.train()
        criterion.train()
        train_sums = defaultdict(float)

        # Freeze backbone for the first freeze_backbone_epochs epochs
        backbone_mult = 0.0 if float(epoch) < freeze_backbone_epochs else 1.0
        epoch_head_lr = float(optimizer.param_groups[0]["lr"])
        epoch_backbone_lr = epoch_head_lr * lr_ratio * backbone_mult
        optimizer.param_groups[1]["lr"] = epoch_backbone_lr

        for step, (images, targets) in enumerate(train_loader, start=1):
            global_step += 1

            # Linear warmup during the first warmup_steps steps
            if warmup_steps > 0 and global_step <= warmup_steps:
                w = float(global_step) / float(warmup_steps)
                optimizer.param_groups[0]["lr"] = epoch_head_lr * w
                optimizer.param_groups[1]["lr"] = epoch_backbone_lr * w
            else:
                optimizer.param_groups[0]["lr"] = epoch_head_lr
                optimizer.param_groups[1]["lr"] = epoch_backbone_lr

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = compute_total_loss(loss_dict, weight_dict)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            train_sums["total_loss"] += float(total_loss.item())
            for key, value in loss_dict.items():
                train_sums[key] += float(value.item())

            if step % max(1, len(train_loader) // 10) == 0 or step == len(train_loader):
                epoch_pbar.set_postfix(
                    epoch=epoch + 1,
                    iter=f"{step}/{len(train_loader)}",
                    lr=f'{optimizer.param_groups[0]["lr"]:.2e}',
                    total=f'{train_sums["total_loss"] / step:.3f}',
                    ce=f'{train_sums["loss_ce"] / step:.3f}',
                    bbox=f'{train_sums["loss_bbox"] / step:.3f}',
                )

        print("total_loss", total_loss)

        if (epoch + 1) % eval_every == 0:
            eval_results = evaluate_detr(model, print_results=False, **eval_kwargs)
            eval_loss_dict = evaluate_test_loss(model, criterion, test_loader)
            epoch_map = float(eval_results["map"])
            epoch_test_loss = float(eval_loss_dict["total_loss"])
            final_map = epoch_map
            final_test_loss = epoch_test_loss
            tqdm.write(
                f"Epoch {epoch + 1}: test_mAP={epoch_map:.4f}, test_loss={epoch_test_loss:.4f}"
            )

        train_losses.append(total_loss.item())
        valid_losses.append(epoch_test_loss)
        valid_maps.append(epoch_map)
        torch.save(
            {
                "train_loss": (counter, valid_losses),
                "valid_loss": (counter, train_losses),
                "valid_mAP": (counter, valid_maps),
            },
            results_dir / f"res_detr",
        )

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), run_dir / f"detector_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), run_dir / "detector_last.pth")

        scheduler.step()

        epoch_pbar.set_postfix(
            epoch=epoch + 1,
            total=f'{train_sums["total_loss"] / len(train_loader):.3f}',
            ce=f'{train_sums["loss_ce"] / len(train_loader):.3f}',
            bbox=f'{train_sums["loss_bbox"] / len(train_loader):.3f}',
        )

        # break

    training_seconds = time.time() - start_time
    print(f"Finished training in {training_seconds:.1f}s")
    print(f"Final test mAP: {final_map:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")

    results_fname = results_dir / f"res_detr"
    plot(results_fname)

    from src.predict import predict_image_detr

    def load_checkpoint_for_inference(checkpoint_path: Path):
        inference_model = SimpleDETR(
            num_classes=len(VOC_CLASSES),
            backbone=backbone,
            num_queries=num_queries,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)
        inference_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        inference_model.eval()
        return inference_model

    checkpoint_path = run_dir / "detector_last.pth"

    inference_model = load_checkpoint_for_inference(checkpoint_path)
    for i in range(10):
        image_name = random.choice(test_dataset.fnames)
        image_bgr = cv2.imread(str(file_root_test / image_name))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Play around with the confidence threshold to see how they affect the predictions.
        detections = predict_image_detr(
            model=inference_model,
            image_name=image_name,
            root_img_directory=str(file_root_test),
            conf_threshold=0.05,
        )

        canvas = image_rgb.copy()
        for det in detections:
            class_idx = VOC_CLASSES.index(det.class_name)
            color = tuple(int(c) for c in COLORS[class_idx])
            x1, y1, x2, y2 = det.box.astype(int).tolist()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.score:.2f}"
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x1
            text_y = max(h + 5, y1)
            cv2.rectangle(
                canvas,
                (text_x, text_y - h - baseline),
                (text_x + w, text_y),
                color,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                label,
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        plt.imsave(results_dir / f"{image_name}", canvas)

        # plt.figure(figsize=(12, 12))
        # plt.imshow(canvas)
        # plt.title(f"DETR Predictions for {image_name}")
        # plt.axis("off")
        # plt.show()
