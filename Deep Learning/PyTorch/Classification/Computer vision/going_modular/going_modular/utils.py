"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import torch
from torch import nn
import numpy as np
import torchvision
import torchvision.models as models
import random
from tqdm.auto import tqdm
from PIL import Image
from timeit import default_timer as timer 
from timm.data import create_transform
import timm



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

    # Get the min test loss and max test acc
    min_test_loss_index = np.argmin(test_loss)
    max_test_acc_index = np.argmax(test_accuracy)
    min_test_loss = test_loss[min_test_loss_index]
    max_test_acc = test_accuracy[max_test_acc_index]

    loss_label = f"Best epochs: {min_test_loss_index} | Min test loss: {min_test_loss:.3f}"
    accuracy_label = f"Best epochs: {max_test_acc_index} | Max test acc: {max_test_acc:.3f}"

    # Setup a plot 
    plt.figure(figsize=(15, 7))
    plt.style.use('fivethirtyeight')

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.scatter(min_test_loss_index, min_test_loss, s=150, color='blue', label=loss_label)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.scatter(max_test_acc_index, max_test_acc, s=150, color='blue', label=accuracy_label)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


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


def pred_and_store(paths: list,
                   model: torch.nn.Module,
                   class_names,
                   transform: torchvision.transforms,
                   device: str) -> List[Dict]:
    '''
    Makes predictions on a list of image paths and stores the results in a list of dictionaries.

    Args:
        paths: A list of image paths.
        model: A PyTorch model.
        class_names: A list of class names for the model.
        transform: A PyTorch transform.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A list of dictionaries containing the image path, class name, prediction probability, prediction class, prediction time
        and whether the prediction was correct.
    '''
    pred_list = []
    for path in tqdm(paths):
        predict_dict = {}
        predict_dict["image_path"] = path
        class_name = path.parent.stem
        predict_dict["class_names"] = class_name
        start = timer()
        img = Image.open(path)
        img_transformed = transform(img).to(device)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            logit = model(img_transformed.unsqueeze(0)).to(device)
        pred_prob = torch.softmax(logit, dim=1)
        pred_class = torch.argmax(pred_prob, dim=1)
        predict_dict["pred_prob"] = f"{round(pred_prob[0][pred_class].item()*100, 2)} %"
        predict_dict["pred_class"] = class_names[pred_class.cpu()]
        stop = timer()
        predict_dict["time_for_pred"] = round(stop - start, 4)
        predict_dict["correct"] = class_name == predict_dict["pred_class"]
        pred_list.append(predict_dict)
    return pred_list


def random_paths(dataset_dir: str,
                 k: int = 3):
    """
    Randomly select path of image from the dataset directy.

    Args: 
        path (string): path to the dataset directory.
        k (int, optional): number of random image path to return.

    Returns:
        List of images path.
    """
    plant_img_path = list(Path(dataset_dir).glob("*/*.jpg"))
    return [Path(filepath) for filepath in random.sample(plant_img_path, k=3)]


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


def create_torchvision_model(num_classes: int = 3,
                             seed: int = 42,
                             model_name: str = "ViT_B_16",
                             weights_name: str = "IMAGENET1K_SWAG_E2E_V1") -> Tuple[nn.Module, torchvision.transforms.Compose]:
    """
    Create a torchvision feature extractor model and transforms

    Args:
        num_classes (int, optional): number of classes. Defaults to 3.
        seed (int, optional): random seed. Defaults to 42.
        model_name (str, optional): name of the model. Defaults to vit_b_16.
        weights_name (str, optional): name of the weights corresponding to the model. Defaults to IMAGENET1K_SWAG_E2E_V1.

    Returns:
        model (nn.Module): torchvision feature extractor model, 
        transforms (torchvision.transforms.Compose): torchvision images transforms
    """
    # Get the model weights
    weights_class_name = model_name + "_Weights"
    weights_class = getattr(models, weights_class_name)
    weights = getattr(weights_class, weights_name)
    # Get automatic transforms from pretrained ViT weights
    transforms = weights.transforms()
    # Get the model architecture with pretrained weights
    model = getattr(models, model_name.lower())(weights=weights)
    # Freeze the feature extractor
    for param in model.parameters():
        param.requires_grad = False 
    # Get its head
    torch.manual_seed(seed)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name   # we keep the last nn.linear
    # last_name is a path, ex : "heads.head"
    # Get the parent module
    parts = last_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    # Replace final layer
    old = getattr(parent, parts[-1])
    setattr(parent, parts[-1], nn.Linear(old.in_features, num_classes))
    return model, transforms


def create_timm_model(model_name: str,
                      class_names: list,
                      IMAGE_SIZE: int = 224,
                      weights_std: str = "IMAGENET_DEFAULT_STD",
                      weights_mean: str = "IMAGENET_DEFAULT_MEAN") -> Tuple[timm.create_model, timm.data.create_transform]:
    """
    Create a timm feature extractor model and transforms

    Args:
        model_name (str): name of the model.
        class_names (list): list of class names.
        weights_std (str, optional): name of the weights standard deviation corresponding to the model. Defaults to IMAGENET_DEFAULT_STD.
        weights_mean (str, optional): name of the weights mean corresponding to the model. Defaults to IMAGENET_DEFAULT_MEAN.

    Returns:
        model (nn.Module): timm feature extractor model, 
        transforms (torchvision.transforms.Compose): timm images transforms
    """
    model = timm.create_model(model_name, 
                              pretrained = True, 
                              num_classes = len(class_names))
    # Create the transform
    transforms = create_transform(input_size=(3, IMAGE_SIZE, IMAGE_SIZE),
                                  mean=model.default_cfg.get("mean", getattr(timm.data.constants, weights_mean)),
                                  std=model.default_cfg.get("std", getattr(timm.data.constants, weights_std)),
                                  crop_pct=model.default_cfg.get("crop_pct", 1.0),
                                  interpolation=model.default_cfg.get("interpolation", "bilinear")
                                 )
    return model, transforms
