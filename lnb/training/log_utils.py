"""Log utilities."""
from typing import Tuple


def get_time_log(
    current_t: float,
    start_t: float,
    epoch_start_t: float,
    i_batch: int,
    epoch: int,
    n_batch: int,
    n_epochs: int,
) -> Tuple[str, str]:
    """Get time logs during training."""
    eta_ep = int((current_t - epoch_start_t) / i_batch * (n_batch - i_batch))
    eta_ep_str = f"{eta_ep // 3600}h {(eta_ep%3600) // 60}m {eta_ep % 60}s"
    eta = eta_ep + int(
        (current_t - start_t)
        / (i_batch + epoch * n_batch)
        * (n_epochs - epoch - 1)
        * n_batch,
    )
    eta_str = f"{eta // 3600}h {(eta%3600) // 60}m {eta % 60}s"
    return eta_str, eta_ep_str
