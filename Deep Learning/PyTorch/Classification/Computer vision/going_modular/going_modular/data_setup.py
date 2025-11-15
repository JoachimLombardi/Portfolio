from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int
):
    '''
    Creates train and test dataloaders

    takes in a training directory and testing directory path and turns them into
    Pytorch Datasets and then into PyTorch Dataloaders.

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms to perform on training and testing data
        batch_size: size of each image batch
        num_workers: number of subprocesses to use for data loading

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Either 'pizza', 'steak', 'sushi'
    '''
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    # Get class names as a list
    class_names = train_data.classes
    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True,
                                pin_memory=True) # put data in pinned memory for faster transfer

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                shuffle=False,
                                pin_memory=True) 
    return train_dataloader, test_dataloader, class_names
