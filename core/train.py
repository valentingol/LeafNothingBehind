import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))

import argparse
import os

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb


from dataset import LNBDataset
from resunet import ExtremeResUnet

import metrics

PATH = "E:\Antoine\Compétitions\Leaf Nothing Behind\\assignment-2023\\assignment-2023"


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train():
    ########################## Wandb & General Constants ##########################

    wandb.init(project="Leaf Nothing Behind")

    name = "lfb"
    checkpoint_dir = os.path.join(PATH, "checkpoints")

    ########################## Model ##########################

    model = ExtremeResUnet(10).cuda()

    ########################## Criterion & Optimzier ##########################

    criterion = metrics.BCEDiceLoss()

    learning_rate = 8e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ########################## Dataset ##########################

    batch_size = 8

    dataset = LNBDataset(PATH, stackify=True)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn
    )

    ########################## Training ##########################

    start_epoch = 0
    num_epochs = 100

    step = 1
    logging_step = 1
    validation_step = 250

    best_loss = 9999

    for epoch in range(start_epoch, num_epochs):
        print("#" * 40)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("#" * 40)

        # step the learning rate scheduler
        lr_scheduler.step()

        # Metric trackers
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        # Batch loop
        loader = tqdm(train_dataloader, desc="Training")
        for idx, (samples, predictions) in enumerate(loader):
            samples = samples.to("cuda")
            predictions = predictions.to("cuda")

            optimizer.zero_grad()

            outputs = model(samples)

            loss = criterion(outputs, predictions)

            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, predictions), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            if step % logging_step == 0:
                wandb.log(
                    {"train_accuracy": train_acc.avg, "train_loss": train_loss.avg}
                )

                loader.set_description(
                    "Training Loss: {:.4f} Acc: {:.4f}".format(
                        train_loss.avg, train_acc.avg
                    )
                )

            # Validation
            if step % validation_step == 0:
                valid_metrics = validation(val_dataloader, model, criterion, step)

                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                # store best loss and save a model checkpoint
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(valid_loader, model, criterion, step):
    # Validation metric trackers
    val_acc = metrics.MetricTracker()
    val_loss = metrics.MetricTracker()

    model.eval()

    loader = tqdm(valid_loader, desc="validation")

    for idx, (sample, target) in enumerate(loader):
        sample = sample.cuda()
        target = target.cuda()

        outputs = model(sample)

        loss = criterion(outputs, target)

        val_acc.update(metrics.dice_coeff(outputs, target), outputs.size(0))
        val_loss.update(loss.data.item(), outputs.size(0))

        if idx == 0:
            imgs = sample.cpu().detach().numpy().squeeze()[1, :]
            target = target.cpu().detach().numpy().squeeze()[1, :]
            pred = outputs.cpu().detach().numpy().squeeze()[1, :]

            print(imgs.shape, target.shape, pred.shape)
            image_samples = [
                wandb.Image(imgs[i, :, :], caption=f"Input at step {step}")
                for i in range(imgs.shape[0])
            ]
            image_target = wandb.Image(target, caption=f"Target at step {step}")
            image_pred = wandb.Image(pred, caption=f"Prediction at step {step}")

            wandb.log(
                {
                    "images_samples": image_samples,
                    "images_result": [image_target, image_pred],
                }
            )
    wandb.log({"accuracy": val_acc.avg, "loss": val_loss.avg})

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(val_loss.avg, val_acc.avg))
    model.train()
    return {"valid_loss": val_loss.avg, "valid_acc": val_acc.avg}


if __name__ == "__main__":
    train()
