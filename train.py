import os
import yaml
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from easydict import EasyDict

from modules.trainer import Trainer
from networks.setup import initialize_models
from data.dataloader import get_pre_transformations, get_default_transforms, pre_transform_dataset, initialize_dataloaders
from helpers.tensorboard_helper import TBoard

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    # Open config file
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Set seed to ensure reproducibility
    set_seed(config.seed)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract ROI dims
    roi_values = config.dataset.roi.split(":")
    width, height, x_offset, y_offset = map(int, roi_values)

    # Load default transforms
    default_transforms = get_default_transforms(config=config, roi_dims=(width, height, x_offset, y_offset))

    # Pre apply transformations to speed up
    if config.dataset.pre_transform:
        dataset_path = Path(config.dataset.path)
        new_dataset_path = dataset_path.with_name(f"{dataset_path.stem}-w{width}-h{height}-x{x_offset}-y{y_offset}")

        pre_transforms = get_pre_transformations(config=config, roi_dims=(width, height, x_offset, y_offset))
        pre_transform_dataset(input_path=config.dataset.path, output_path=new_dataset_path, transforms=pre_transforms)

        config.dataset.path = str(new_dataset_path)
        default_transforms = get_default_transforms(config=config, roi_dims=None)

    train_loader, validation_loader, test_set = initialize_dataloaders(config=config, default_transforms=default_transforms)

    # Load models
    teacher, student, autoencoder = initialize_models(config=config, with_print=True)

    # Setup TBoard dir
    tboard_log_path = os.path.join("logs", os.path.basename(config.saver.exp_path))
    os.makedirs(tboard_log_path, exist_ok=True)
    tboard = TBoard(log_path=tboard_log_path)

    trainer = Trainer(
        config=config, teacher=teacher, student=student, autoencoder=autoencoder, 
        train_loader=train_loader, validation_loader=validation_loader, test_set=test_set, 
        default_transforms=default_transforms, device=device, tboard=tboard)
    
    trainer.train()

if __name__ == '__main__':
    main()