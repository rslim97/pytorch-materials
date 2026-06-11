import os
import torch.utils
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision import datasets, models

import torch.optim as optim

from model.resnet import resnet18, resnet34, resnet50, resnet101
from model.vit import VisionTransformer


torch.manual_seed(34)


def plot(results_fname):
    plt.figure(figsize=(10, 5))
    # Load saved data
    x = torch.load(results_fname, weights_only=True)
    train_counter, train_losses = x["train_loss"]
    valid_counter, valid_losses = x["val_loss"]
    val_accs = x["val_accs"]
    plt.plot(
        train_counter,
        train_losses,
        zorder=+100,
        color="#1f77b4",
        label="Train loss",
    )
    plt.scatter(
        valid_counter,
        valid_losses,
        zorder=+100,
        color="#ff7f0e",
        label="Val loss",
    )
    plt.scatter(
        [valid_counter[np.argmin(valid_losses)]],
        [min(valid_losses)],
        color="black",
        zorder=+200,
        label=(f"Best Accuracy: {max(val_accs)*100:.2f}%"),
    )
    plt.title(results_fname)
    plt.xlabel("Number of Examples Seen by the model")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(results_fname + ".png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("number of cuda devices:", torch.cuda.device_count())

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(proj_dir, "dataset")
    input_size = 224
    batch_size = 64
    load_pretrain = True
    ckpt_dir = os.path.join(proj_dir, "checkpoints")
    results_dir = os.path.join(proj_dir, "results")
    lr = 1e-5
    n_epochs = 5
    bool_train = True

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("load_pretrain", load_pretrain)
    print("lr", lr)
    print("n_epochs", n_epochs)

    # mean = torch.tensor([0.4767, 0.4616, 0.4160])
    # std = torch.tensor([0.2284, 0.2287, 0.2540])
    data_transforms = {
        "train": T.Compose(
            [
                T.RandomResizedCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Normalize(mean, std),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.ToTensor(),
                # T.Normalize(mean, std),
            ]
        ),
    }

    train_data = datasets.ImageFolder(
        os.path.join(data_dir, "train"), data_transforms["train"]
    )
    testval_data = datasets.ImageFolder(
        os.path.join(data_dir, "test"), data_transforms["test"]
    )
    test_len = int(len(testval_data) * 0.5)
    val_len = len(testval_data) - test_len
    print(f"Using {test_len} samples for testing and {val_len} for validation.")

    test_data, val_data = torch.utils.data.random_split(
        testval_data, [test_len, val_len], generator=torch.Generator()
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=1
    )

    # Visualize a batch of training data
    import matplotlib.pyplot as plt

    # Obtain one batch of training images
    images, labels = next(iter(train_loader))  # Labels are 0 to c-1,
    # where c is the total number of classes
    images = images.numpy()
    print(images.shape)  # (H: 224, W: 224)
    print(labels.shape)

    # Plot the images in the batch, along withe the corresponding labels
    fig, subs = plt.subplots(2, batch_size // 2, figsize=(25, 4))
    for idx, sub in zip(np.arange(batch_size), subs.flatten()):
        sub.imshow(images[idx].transpose(1, 2, 0))
        sub.set_title(str(labels[idx].item()))
        sub.axis("off")
    plt.savefig(os.path.join(results_dir, "CIFAKE-data.png"))
    plt.show()

    fname = "adam_cifake_fc"
    ckpt_fname = os.path.join(ckpt_dir, fname + ".pt")

    criterion = nn.CrossEntropyLoss()

    model = resnet50(num_classes=2, pretrained=load_pretrain)
    # model = VisionTransformer(
    #     num_classes=2, img_size=224, patch_size=16, dropout=0.2, num_layers=6
    # )

    # scripted = torch.jit.script(model)
    # torch.jit.save(scripted, 'resnet50-network.pt')

    # if load_pretrain:
    #     load_pretrained(model)

    model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)
    print(model)

    # Pre-training sanity check
    test_loss = 0.0
    test_acc = 0.0
    images, labels = next(iter(train_loader))  # Labels are 0 to c-1,
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)
        # Use a single batch for pre-training check
        output = model(images)  # (N, C)

        loss = criterion(output, labels)

        batch_size = images.size(0)

        test_loss += loss.item() * batch_size  # loss.item() returns avg over batch_size
        test_acc += torch.sum(torch.argmax(output, dim=1) == labels).item()
        test_loss /= batch_size
        test_acc /= batch_size
        print()
        print(f"Pre-training loss: {test_loss:.6f}\n")
        print(f"Pre-training acc: {test_acc:.5f}\n")

        # Visualize sample test results
        images, labels = next(iter(test_loader))

        images = images.to(device)
        labels = labels.to(device)
        # Get sample outputs
        output = model(images)

        # Convert output probabilities to predicted class
        indices, preds = torch.max(output, dim=1)
        # Convert to numpy
        images = images.detach().cpu().numpy()

        # Plot the images in the batch, along with predicted and true labels
        fig, subs = plt.subplots(2, 10, figsize=(25, 4))
        for idx, sub in zip(range(20), subs.flatten()):
            sub.imshow(images[idx].transpose(1, 2, 0))
            # sub.imshow(np.squeeze(images[idx]), cmap="gray")
            sub.set_title(
                f"pred:{str(preds[idx].item())} label:{str(labels[idx].item())}",
                color=("green" if preds[idx] == labels[idx] else "red"),
            )
            sub.axis("off")
        plt.savefig(os.path.join(results_dir, "CIFAKE-pre-training.png"))
        plt.show()

    # Training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

    val_loss_min = np.float64("inf")

    if bool_train:
        # Model training
        model.train()
        if load_pretrain:
            # Freeze layers up through layer 1
            for name, layer in model.named_children():
                if name in ["conv1", "bn1", "layer1"]:
                    for param in layer.parameters():
                        param.requires_grad = False
                    for m in layer.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
        # Print layer freezing status
        for name, param in model.named_parameters():
            print(f"{name} requires_grad={param.requires_grad}")
        counter = [i * len(train_loader.dataset) for i in range(n_epochs)]
        train_losses = []
        val_losses = []
        val_accs = []
        for epoch in range(n_epochs):
            train_loss = 0.0
            val_loss = 0.0
            val_acc = 0.0
            for batch_idx, (img, label) in tqdm.tqdm(
                iterable=enumerate(train_loader),
                desc="training",
                total=len(train_loader),
                leave=True,
                ncols=80,
            ):
                optimizer.zero_grad()

                img = img.to(device)
                label = label.to(device)

                pred = model(img)

                loss = criterion(pred, label)

                loss.backward()

                optimizer.step()

                batch_size = img.size(0)

                train_loss += (
                    loss.item() * batch_size
                )  # loss.item() returns avg over batch_size

                del img, label, pred
                torch.cuda.empty_cache()

            # Get training loss over one epoch.
            # Note: num_samples = num_batches * batch_size
            train_loss = train_loss / len(train_loader.dataset)

            # Validation to determine when to stop our model training
            with torch.no_grad():
                model.eval()

                for batch_idx, (img, label) in tqdm.tqdm(
                    iterable=enumerate(val_loader),
                    desc="validation",
                    total=len(val_loader),
                    leave=True,
                    ncols=80,
                ):
                    img = img.to(device)
                    label = label.to(device)

                    pred = model(img)

                    loss = criterion(pred, label)

                    # Compute running/moving average of validation loss
                    val_loss += (loss.item() - val_loss) / (batch_idx + 1)
                    val_acc += torch.sum(torch.argmax(pred, dim=1) == label).item()

                    del img, label, pred
                    torch.cuda.empty_cache()

                if val_loss <= val_loss_min:
                    print(
                        "Validation loss decreased ({:.5f}->{:.5f}). Saving model".format(
                            val_loss_min, val_loss
                        )
                    )
                    torch.save(model.state_dict(), ckpt_fname)
                    val_loss_min = val_loss

            val_acc /= len(val_loader.dataset)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            results_fname = os.path.join(results_dir, f"res_{fname}")
            torch.save(
                {
                    "val_loss": (counter, val_losses),
                    "train_loss": (counter, train_losses),
                    "val_accs": val_accs,
                },
                results_fname,
            )
            print(
                f"Epoch {epoch+1}: training loss {train_loss:.5f}, valid_loss {val_loss:.5f}"
            )

    else:
        # Load saved model
        model.load_state_dict(torch.load(ckpt_fname, weights_only=True))
    plot(results_fname)

    # Testing
    test_loss = 0.0
    test_acc = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (img, label) in tqdm.tqdm(
            iterable=enumerate(test_loader),
            desc="testing",
            total=len(test_loader),
            leave=True,
            ncols=80,
        ):
            img = img.to(device)
            label = label.to(device)

            pred = model(img)  # (N, num_classes)

            loss = criterion(pred, label)

            test_loss += (
                loss.item() * img.shape[0]
            )  # loss.item() returns avg over batch_size

            test_acc += torch.sum(torch.argmax(pred, dim=1) == label).item()
            # print('test_acc', test_acc)

            del img, label, pred
            torch.cuda.empty_cache()

        test_loss /= len(test_loader.dataset)  # Div by num of samples
        test_acc /= len(test_loader.dataset)

        print("\n")
        print(f"Test loss: {test_loss:.6f}\n")
        print(f"Test acc: {test_acc:.5f}\n")

        # Visualize sample test results
        images, labels = next(iter(test_loader))

        images = images.to(device)
        labels = labels.to(device)

        # Get sample outputs
        output = model(images)

        # Convert output probabilities to predicted class
        indices, preds = torch.max(output, dim=1)
        # Convert to numpy
        images = images.detach().cpu().numpy()

        # Plot the images in the batch, along with predicted and true labels
        fig, subs = plt.subplots(2, batch_size // 2, figsize=(25, 4))
        for idx, sub in zip(range(batch_size), subs.flatten()):
            sub.imshow(images[idx].transpose(1, 2, 0))
            sub.set_title(
                f"pred:{str(preds[idx].item())} label:{str(labels[idx].item())}",
                color=("green" if preds[idx] == labels[idx] else "red"),
            )
            sub.axis("off")
        plt.savefig(os.path.join(results_dir, "CIFAKE-after-training.png"))
        plt.show()
