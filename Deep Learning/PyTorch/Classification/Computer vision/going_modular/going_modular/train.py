import os
import torch
from torch import nn
from torchvision import transforms
from data_import import get_data
from data_setup import create_dataloaders
from engine import train_test_step
from model_builder import TinyVGG
from utils import save_model


if __name__ == "__main__":

   # Setup hyperparameters
  NUM_EPOCHS = 5
  NUM_BLOCKS = 2
  INPUT_SHAPE = 3
  BATCH_SIZE = 32
  HIDDEN_UNITS = 10
  LEARNING_RATE = 0.001

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Get data
  get_data(url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
          file="pizza_steak_sushi")

  # Setup directories
  train_dir = "data/pizza_steak_sushi/train"
  test_dir = "data/pizza_steak_sushi/test"


  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Create model with help from model_builder.py
  model = TinyVGG(
      input_shape=INPUT_SHAPE,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names),
      num_blocks=NUM_BLOCKS
  ).to(device)

  # Set loss and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Start training with help from engine.py
  results = train_test_step(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=NUM_EPOCHS,
            device=device)

  # Save the model with help from utils.py
  save_model(model=model,
            target_dir="models",
            model_name="05_going_modular_script_mode_tinyvgg_model.pth")
