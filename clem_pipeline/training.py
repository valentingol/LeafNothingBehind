import torch 
from data.dataloader import TrainDataset
from tqdm import tqdm 
import os 
from clem_pipeline.models.custom_models import get_model
from clem_pipeline.utils import MetricTracker, parse_training_tensor, WeightedMSE


def train_loop(dataloader, model, criterion, optimizer, loss_tracker, device, epoch):
    model.train()

    loader = tqdm(dataloader, desc="training")
    for data, timestamp in loader:
        data.to(device)
        timestamp.to(device)

        t2s1, t1s1, ts1, t2s2, t1s2, ts2, t2s2_mask, t1s2_mask, ts2_binary_mask = parse_training_tensor(data)

        pred = model(t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask, timestamp)

        loss = criterion(pred, ts2, ts2_binary_mask)

        loss_tracker.update(loss.item(), data.size(0))
        loader.set_description(f"Training Loss: {loss_tracker.avg:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val_loop(dataloader, model, criterion, loss_tracker, device, epoch):
    model.eval()

    loader = tqdm(dataloader, desc="validation")
    for data, timestamp in loader:
        data.to(device)
        timestamp.to(device)

        t2s1, t1s1, ts1, t2s2, t1s2, ts2, t2s2_mask, t1s2_mask, ts2_binary_mask = parse_training_tensor(data)

        with torch.no_grad():
            pred = model(t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask)
            loss = criterion(pred, ts2, ts2_binary_mask)

        loss_tracker.update(loss.item(), data.size(0))
        loader.set_description(f"Validation Loss: {loss_tracker.avg:.4f}")
    
def get_default_run_name():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")



def get_optimizer(parameters, model):
    match parameters["optimizer"]:
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=parameters["lr"])
        
        case "sgd":
            return torch.optim.SGD(model.parameters(), lr=parameters["lr"], momentum=parameters["momentum"])

def get_lr_scheduler(
    parameters, optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler:
    return (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=parameters["lr_decay_epochs"],
            gamma=parameters["lr_decay"],
        )
        if parameters["use_lr_decay"]
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    )

if __name__ == "__main__":
    from clem_pipeline.models.custom_models import ModelType

    use_wandb = False
    
    run_name = None
    weight_checkpoint_freq = 0 # 0 means no checkpoint

    parameters = {
        "model_type": ModelType.S1_MASK_IDEA2,

        "optimizer": "adam", #  "adam" or "sgd"
        "momentum": 0.9, # only used for sgd

        "lr": 1e-3,
        "use_lr_decay": True,
        "lr_decay": 0.5,
        "lr_decay_epochs": [10, 20, 30],

        "epochs": 10,
        "batch_size": 4,

        "num_workers": 4,
        "pin_memory": True,

        "saving_folder": "saved_weights",

        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if use_wandb:
        import wandb

        wandb.init(project="PROJECT_NAME", config=parameters)

        parameters = wandb.config # for sweep

        run_name = wandb.run.name

    if run_name is None:
        run_name = get_default_run_name()

    print(f"Using {parameters['device']} device")

    dataset = TrainDataset(dataset_path='../assignment-2023', csv_data='image_series.csv',
                            csv_grid='../assignment-2023/square.csv', grid_augmentation=True, all_masks=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=parameters["batch_size"], shuffle=True,
                                                num_workers=parameters["num_workers"], pin_memory=parameters["pin_memory"])
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=parameters["batch_size"], shuffle=False,
                                                num_workers=parameters["num_workers"], pin_memory=parameters["pin_memory"])

    model = get_model(parameters).to(parameters['device'])
    criterion = WeightedMSE()
    optimizer = get_optimizer(parameters, model)
    lr_scheduler = get_lr_scheduler(parameters, optimizer)

    train_loss_tracker = MetricTracker()
    val_loss_tracker = MetricTracker()
    best_val_loss = float("inf")

    for epoch in range(parameters['epochs']):
        train_loss_tracker.reset()
        val_loss_tracker.reset()

        train_loop(train_dataloader, model, criterion, optimizer, train_loss_tracker, parameters['device'], epoch)
        val_loop(val_dataloader, model, criterion, val_loss_tracker, parameters['device'], epoch)

        if weight_checkpoint_freq and (epoch+1) % weight_checkpoint_freq == 0:
            torch.save(model.state_dict(), os.path.join(parameters["saving_folder"], f"{run_name}_epoch_{epoch+1}.pth"))
        
        if val_loss_tracker.avg < best_val_loss:
            best_val_loss = val_loss_tracker.avg
            torch.save(model.state_dict(), os.path.join(parameters["saving_folder"], f"{run_name}_best.pth"))

        if use_wandb:
            wandb.log({"train_loss": train_loss_tracker.avg, "val_loss": val_loss_tracker.avg, "best_val_loss": best_val_loss})
        
        lr_scheduler.step()
