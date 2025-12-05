"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import torch
from torch import nn
import numpy as np
import torchvision


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)


def load_model(model_path: str,
               model_builder: torch.nn.Module,
               device: torch.device,
               params: dict = None):

    '''
    Loads a PyTorch model from a target directory.

    Args:
    model_path: A directory for saving the model to.
    model_builder: A model builder to use to create the model.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    params: A dictionary of parameters to include when creating the model
        input_shape: Number of channels in the input
        output_shape: Number of channels in the output
        hidden_units: Number of hidden units
        num_blocks: Number of convolutional layers

    Returns:
    A PyTorch model from a target directory
    '''
    if params:
      model_builder = model_builder(**params)
    model_builder.load_state_dict(torch.load(model_path, map_location=device))
    return model_builder


def get_classes(image_dir):

  train_path = Path(image_dir)
  class_names = [d.name for d in train_path.iterdir() if d.is_dir()]
  return class_names


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.
    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)  # Create log directory path
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def img_denorm(tensor_image) -> np.ndarray:
    '''
    Denormalizes an image tensor.

    Args:
        tensor_image: An image tensor.

    Returns:
        A denormalized image tensor.
    '''
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor_image = tensor_image * std + mean
    tensor_image = tensor_image.permute(1,2,0)
    tensor_image = np.clip(tensor_image,0,1)
    return tensor_image


def freeze_pretrained_model(model, class_names, device):
    '''
    Freezes the pretrained model's parameters.

    Args:
        model: A pretrained PyTorch model.
        class_names: A list of class names for the model.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    '''
    for params in model.parameters():
        params.requires_grad = False

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            in_features = module.in_features 
            break

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=len(class_names))
    ).to(device)


def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=42):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be split? 
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        seed (int, optional): Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and 
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    if split_size is None:
        return dataset, None

    if split_size < 0 or split_size > 1:
        raise ValueError("split_size must be between 0 and 1")

    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size) # desired length
    length_2 = len(dataset) - length_1 # remaining length

    # Print out info
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")

    # Create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(dataset, 
                                                                   lengths=[length_1, length_2],
                                                                   generator=torch.manual_seed(seed)) # set the random seed for reproducible splits
    return random_split_1, random_split_2
