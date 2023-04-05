import torch 
from torch import nn 
from data.dataloader import TrainDataset


class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, pred, target, weight):
        return torch.mean(weight * (pred - target) ** 2)
    

def train_loop(dataloader, model, criterion, optimizer, loss_tracker, device):
    model.train()

    for data, timestamp in dataloader:
        data.to(device)
        timestamp.to(device)

        t2s1, t1s1, ts1, t2s2, t1s2, ts2, t2s2_mask, t1s2_mask, ts2_binary_mask = parse_training_tensor(data)

        pred = model(t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask)

        loss = criterion(pred, ts2, ts2_binary_mask)

        loss_tracker.update(loss.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val_loop(dataloader, model, criterion, loss_tracker, device):
    model.eval()

    for data, timestamp in dataloader:
        data.to(device)
        timestamp.to(device)

        t2s1, t1s1, ts1, t2s2, t1s2, ts2, t2s2_mask, t1s2_mask, ts2_binary_mask = parse_training_tensor(data)

        with torch.no_grad():
            pred = model(t2s1, t1s1, ts1, t2s2, t1s2, t2s2_mask, t1s2_mask)
            loss = criterion(pred, ts2, ts2_binary_mask)

        loss_tracker.update(loss.item(), data.size(0))

if __name__ == "__main__":
    from clem_pipeline.utils import parse_training_tensor
    from clem_pipeline.models.custom_models import OnlyS1Idea1

    use_wandb = False

    parameters = {
        "batch_size": 4,
        "lr": 1e-3,
        "epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if use_wandb:
        import wandb

        wandb.init(project="PROJECT_NAME", entity="clem", config=parameters)

        parameters = wandb.config # for sweep


    print(f"Using {parameters['device']} device")

    dataset = TrainDataset(dataset_path='../assignment-2023', csv_data='image_series.csv',
                            csv_grid='../assignment-2023/square.csv', grid_augmentation=True, all_masks=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                                num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                                num_workers=4)

    model = OnlyS1Idea1().to(parameters['device'])
    criterion = WeightedMSE()
    train_loss_tracker = MetricTracker()
    val_loss_tracker = MetricTracker()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])

    best_val_loss = float("inf")

    for epoch in range(parameters['epochs']):

        print(f"Epoch {epoch}")
        train_loss_tracker.reset()
        val_loss_tracker.reset()

        train_loop(train_dataloader, model, criterion, optimizer, train_loss_tracker, parameters['device'])
        val_loop(val_dataloader, model, criterion, val_loss_tracker, parameters['device'])

        
        if val_loss_tracker.avg < best_val_loss:
            best_val_loss = val_loss_tracker.avg
            # save model ?

        # wandb
        print(f"ep:{epoch}/{parameters['epochs']}| train loss: {train_loss_tracker.avg:.4f} | val loss: {val_loss_tracker.avg:.4f} | best val loss: {best_val_loss:.4f}")

