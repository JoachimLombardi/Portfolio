from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from going_modular.going_modular.utils import split_dataset
from typing import Tuple, List
import torch
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

def create_dataloaders(
        transform: transforms.Compose,
        train_dir: str,
        test_dir: str = None,
        test_transform: transforms.Compose = None,
        batch_size: int = 32,
        num_workers: int = os.cpu_count(),
        split_size: float | None = None,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    '''
    Creates train and test dataloaders
    Takes in a training directory and testing directory path and turns them into Pytorch Datasets 
    and then into PyTorch Dataloaders.
    If test_dir is provided train and test datasets sizes can be reduced to split_size of their sizes if split_size is provided
    or if test_dir is not provided dataset will be split into train and test datasets.
    (split_size of train size for train and (1 - split_size) of train size for test).
    If classes are unbalanced, images frequencies can be weighted by the inverse of their frequencies.
    If test_transform is provided (data augmentation), test_transform will be normal transform and transform will be data augmentation

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms to perform on training and testing data
        batch_size: size of each image batch
        num_workers: number of subprocesses to use for data loading
        split_size: percentage of train and test datasets to use for train and test if test_dir is provided, default is None
        else percentage of dataset to use for train and the rest for test, default is 0.8
        seed: random seed for random number generators, default is 42
        sampler: Defines order and ratio of samples to be retrieved from a given dataset

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Either 'pizza', 'steak', 'sushi'
    '''
    # Use ImageFolder to create train dataset
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    # Get class names as a list
    class_names = train_data.classes
    # if test_dir is provided, use ImageFolder to create test dataset and split it if split_size is provided
    if test_dir:
        test_data = datasets.ImageFolder(test_dir, transform=test_transform or transform) 
        print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
        train_data, _ = split_dataset(train_data, split_size, seed)
        test_data, _ = split_dataset(test_data, split_size, seed)
    # if test_dir is not provided, split train dataset into train and test datasets
    else:
        if not split_size:
            split_size = 0.8
        train_data, test_data = split_dataset(train_data, split_size, seed)
        # If test_transform is provided, apply it to test dataset
        if test_transform:
            test_data.transform = test_transform
    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    labels = [train_data.dataset.samples[i][1] for i in train_data.indices]
    counts = Counter(labels)
    total = sum(counts.values())
    ratios = {cls: count/total for cls, count in counts.items()}
    sampler = None
    if max(ratios.values()) - min(ratios.values()) > 0.1:  # check if classes are balanced
        print("Classes are not balanced, creating weights for each sample...")
        class_weights = {cls: 1/count for cls, count in counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),  # size of train subset
            replacement=True
        )
    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=(sampler is None), # shuffle the data default, don't shuffle if sampler is defined
                                pin_memory=True,  # put data in pinned memory for faster transfer
                                sampler=sampler) # only for train

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                shuffle=False,
                                pin_memory=True) 
    return train_dataloader, test_dataloader, class_names
