from torchvision import transforms
from utils import load_model, get_classes
from model_builder import TinyVGG
from predict import predict_image
import torch

HIDDEN_UNITS = 10
NUM_BLOCKS = 2
INPUT_SHAPE = 3

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# get classes
class_names = get_classes(image_dir="data/pizza_steak_sushi/train")

# Setup transforms
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load model
model = load_model(model_path="models/05_going_modular_script_mode_tinyvgg_model.pth",
                   model_builder=TinyVGG,
                   device=device,
                   input_shape=INPUT_SHAPE,
                   output_shape=len(class_names),
                   hidden_units=HIDDEN_UNITS,
                   num_blocks=NUM_BLOCKS)

# Predict
predict_image(img_path="images/istockphoto-540233806-612x612.jpg",
              model=model,
              class_names=class_names,
              device=device,
              image_transform=image_transform)
