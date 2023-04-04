import torch 

def parse_training_tensor(data: torch.Tensor):
    """
    data (torch.Tensor): shape [batch_size, 3, 14, 256, 256] (LAI, LAI mask, VV, VH, one hot masks (10,))

    returns:
        - t2s1 (torch.Tensor): shape [batch_size, 2, 256, 256]
        - t1s1 (torch.Tensor): shape [batch_size, 2, 256, 256]
        - ts1 (torch.Tensor): shape [batch_size, 2, 256, 256]
        - t2s2 (torch.Tensor): shape [batch_size, 256, 256]
        - t1s2 (torch.Tensor): shape [batch_size, 256, 256]
        - ts2 (torch.Tensor): shape [batch_size, 256, 256]
        - t2s2_mask (torch.Tensor): shape [batch_size, 10, 256, 256]
        - t1s2_mask (torch.Tensor): shape [batch_size, 10, 256, 256]
        - ts2_binary_mask (torch.Tensor): shape [batch_size, 256, 256]
    """
    t2s1 = data[:, 0, 2:4, ...]
    t1s1 = data[:, 1, 2:4, ...]
    ts1 = data[:, 2, 2:4, ...]

    t2s2 = data[:, 0, 0, ...]
    t1s2 = data[:, 1, 0, ...]
    ts2 = data[:, 2, 0, ...]
    
    t2s2_mask = data[:, 0, 4:14, ...]
    t1s2_mask = data[:, 1, 4:14, ...]

    ts2_binary_mask = data[:, 2, 1, ...]

    return t2s1, t1s1, ts1, t2s2, t1s2, ts2, t2s2_mask, t1s2_mask, ts2_binary_mask