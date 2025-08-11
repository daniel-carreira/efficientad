import os
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

class ImageDatasetWithoutTarget(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [f for f in os.listdir(root_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, _ = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, _ = super().__getitem__(index)
        return sample, target, path

class InfiniteDataloader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
        
class CropROI:
    def __init__(self, roi_dims):
        self.crop_width = roi_dims[0]
        self.crop_height = roi_dims[1]
        self.x = roi_dims[2]
        self.y = roi_dims[3]
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img[:, self.y:self.y + self.crop_height, self.x:self.x + self.crop_width]
        else:
            return transforms.functional.crop(img, self.y, self.x, self.crop_height, self.crop_width)

class StepTransforms:
    def __init__(self, transform_pdn, transform_pdn_default, transform_ae, transform_ae_default, transform_steps):
        """
        Args:
            transform_pdn: Full training transform for PDN.
            transform_pdn_default: Default transform for PDN.
            transform_ae: Full training transform for AE.
            transform_ae_default: Default transform for AE.
            transform_steps: Number of steps to apply the full training transforms.
        """
        self.transform_pdn = transform_pdn
        self.transform_pdn_default = transform_pdn_default
        self.transform_ae = transform_ae
        self.transform_ae_default = transform_ae_default
        self.transform_steps = transform_steps
        self.step_counter = 0  # Single step counter for both transforms

    def __call__(self, img):
        """
        Applies the appropriate transforms for PDN and AE based on the current step.
        
        Args:
            img: Input image.
        
        Returns:
            Tuple of transformed images (pdn_transformed, ae_transformed).
        """
        if self.step_counter < self.transform_steps:
            pdn_transformed = self.transform_pdn(img)
            ae_transformed = self.transform_ae(img)
        else:
            pdn_transformed = self.transform_pdn_default(img)
            ae_transformed = self.transform_ae_default(img)

        return pdn_transformed, ae_transformed

    def increment_step(self):
        """Increments the step counter."""
        self.step_counter += 1

def get_pre_transformations(config, roi_dims=None):
    """Constructs data transformations."""
    transform_list = []
    
    if roi_dims is not None:
        transform_list.append(CropROI(roi_dims))
    
    transform_list.extend([
        transforms.Resize((config.net.input_size, config.net.input_size)),
    ])
    return transforms.Compose(transform_list)

def count_files(path):
    """Recursively counts files, following symlinks for directories."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_symlink():
            resolved = entry.resolve()
            if resolved.is_dir():
                total += count_files(resolved)  # Recursively count symlinked directories
            elif resolved.is_file():
                total += 1  # Count symlinked files
        elif entry.is_file():
            total += 1  # Count regular files
    return total

def pre_transform_dataset(input_path, output_path, transforms, desc='[Dataset] Applying transformations...'):
    """
    Pre-crops all images in a dataset to the specified dimensions, preserving the folder structure.

    Args:
        input_path (str): Path to the root directory of the dataset.
        output_path (str): Path to save the cropped dataset.
        transforms (torchvision.transforms): Transformations to apply.
        desc (str): Description shown on the tqdm progress bar.

    Returns:
        None
    """
    dataset_path = Path(input_path)  # Use Path object for dataset path
    output_path = Path(output_path)    # Use Path object for output path

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Walk through the dataset and add a progress bar
    total_files = count_files(dataset_path)
    
    # tqdm for progress tracking, dynamically set the description
    with tqdm(total=total_files, desc=desc, unit="file") as pbar:
        for original_path in dataset_path.rglob('*'):  # Use rglob to search recursively
            # Create the corresponding new path inside output directory
            relative_path = original_path.relative_to(dataset_path)
            new_path = output_path / relative_path

            # Symlink support
            if original_path.is_symlink():
                resolved_path = original_path.resolve()
                if resolved_path.is_dir():
                    # If it's a directory, we need to iterate its contents
                    for file_path in resolved_path.rglob('*'):
                        new_file_path = new_path / file_path.name
                        process_file(file_path, new_file_path, transforms, pbar)

            # Process non-symlink files
            process_file(original_path, new_path, transforms, pbar)


def process_file(original_path, new_path, transforms, pbar):
    """Handles image processing and copying of non-image files."""
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if new_path.exists():
        pbar.update(1)
        return

    if original_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        try:
            with Image.open(original_path) as img:
                transformed_image = transforms(img)
                transformed_image.save(new_path)
        except OSError as e:
            print(f"Error processing {original_path}: {e}")
    
    elif original_path.suffix.lower() in {'.txt'}:
        shutil.copy(original_path, new_path)

    pbar.update(1)

def get_default_transforms(config, roi_dims=None, with_norm=True):
    """Constructs data transformations."""
    transform_list = [transforms.ToTensor()]
    
    if roi_dims is not None:
        transform_list.append(CropROI(roi_dims))
    
    transform_list.append(transforms.Resize((config.net.input_size, config.net.input_size)))

    if with_norm:
        transform_list.append(transforms.Normalize(mean=config.dataset.norm_mean, std=config.dataset.norm_std))

    return transforms.Compose(transform_list)

def initialize_dataloaders(config, default_transforms):
    """Initializes and returns dataloaders."""
    dataset_path = config.dataset.path

    # PDN Transforms
    transform_pdn_list = default_transforms.transforms[:-1]
    if config.dataset.train.crop:
        transform_pdn_list.append(transforms.RandomResizedCrop(size=config.net.input_size, scale=(0.95, 1.05)))

    if config.dataset.train.hflip:
        transform_pdn_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if config.dataset.train.vflip:
        transform_pdn_list.append(transforms.RandomVerticalFlip(p=0.5))

    if config.dataset.train.rotate:
        transform_pdn_list.append(transforms.RandomRotation(degrees=30))

    if config.dataset.train.translation:
        transform_pdn_list.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))

    if config.dataset.train.grayscale:
        transform_pdn_list.append(transforms.RandomGrayscale(p=0.3))

    if config.dataset.train.jitter:
        transform_pdn_list.append(transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ]))

    if config.dataset.train.blur:
        transform_pdn_list.append(transforms.RandomApply(
            transforms=[transforms.GaussianBlur(3)],
            p=0.3
        ))

    transform_pdn_list.append(transforms.Normalize(mean=config.dataset.norm_mean, std=config.dataset.norm_std))
    transform_pdn_full = transforms.Compose(transform_pdn_list)

    # AE Transforms
    transform_ae_list = default_transforms.transforms[:-1]
    transform_ae_list.append(transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2),
    ]))

    transform_ae_list.append(transforms.Normalize(mean=config.dataset.norm_mean, std=config.dataset.norm_std))
    transform_ae_full = transforms.Compose(transform_ae_list)

    # Create a single StepTransforms instance for both PDN and AE
    step_transforms = StepTransforms(
        transform_pdn=transform_pdn_full,
        transform_pdn_default=default_transforms,
        transform_ae=transform_ae_full,
        transform_ae_default=default_transforms,
        transform_steps=config.trainer.transform_steps
    )
    
    # Train & Valid Dataloaders
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, "train"),
        transform=step_transforms,
    )
    full_train_set.step_transforms = step_transforms

    train_size = int(0.95 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    train_set, validation_set = torch.utils.data.random_split(
        full_train_set, [train_size, validation_size], torch.Generator()
    )
    
    train_loader = DataLoader(train_set, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.dataset.workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=config.dataset.batch_size)

    test_set = ImageFolderWithPath(os.path.join(dataset_path, "test"))

    return train_loader, validation_loader, test_set