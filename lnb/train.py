"""Launch training of a model."""
import os
import random

import torch
import wandb

from lnb.training.trainer import run
from lnb.yaecs_config.config import ProjectConfig


def main() -> None:
    """Make a train with wandb."""
    # Set seed
    torch.manual_seed(0)
    random.seed(0)

    yaecs_config = ProjectConfig.build_from_argv()
    config = yaecs_config.get_dict(deep=True)

    # New id (for model name)
    os.makedirs('./.wandb_archives', exist_ok=True)
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        group=config["archi_name"],
        config=config,
        dir='./.wandb_archives',
    )
    run(dict(wandb.config))
    wandb.finish()


if __name__ == '__main__':
    main()
