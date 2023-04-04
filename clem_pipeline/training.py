import torch 
from data.dataloader import TrainDataset



def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    from clem_pipeline.utils import parse_training_tensor

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb = False

    print(f"Using {DEVICE} device")

    dataset = TrainDataset(dataset_path='../assignment-2023', csv_data='image_series.csv',
                            csv_grid='../assignment-2023/square.csv', grid_augmentation=True, all_masks=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                                num_workers=4)
    

    for data, timestamp in dataloader:
        print(data.shape)
        print(timestamp.shape)
        for element in parse_training_tensor(data):
            print(element.shape)
            
        break    
