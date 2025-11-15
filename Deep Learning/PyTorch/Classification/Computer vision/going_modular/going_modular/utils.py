"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

import torch

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
               input_shape: int,
               output_shape: int,
               hidden_units: int,
               num_blocks: int):

    '''
    Loads a PyTorch model from a target directory.

    Args:
    model_path: A directory for saving the model to.
    model_builder: A model builder to use to create the model.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    input_shape: Number of channels in the input
    output_shape: Number of channels in the output
    hidden_units: Number of hidden units
    num_blocks: Number of convolutional layers

    Returns:
    A PyTorch model from a target directory
    '''
    loaded_model = model_builder(input_shape=input_shape,
                                 hidden_units=hidden_units,
                                 output_shape=output_shape,
                                 num_blocks=num_blocks
                                ).to(device)
    loaded_model.load_state_dict(torch.load(model_path))
    return loaded_model


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
    plt.legend();
