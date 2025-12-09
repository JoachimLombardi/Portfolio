### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
from timeit import default_timer as timer
from typing import Tuple, Dict
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm

# Setup class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

### 2. Model and transforms preparation ###

# Create the ViTB16 model
rexnet_150_model = timm.create_model("rexnet_150", 
                                     pretrained = True, 
                                     num_classes = len(class_names))

# Create the transform
rexnet_150_transforms = create_transform(input_size=rexnet_150_model.default_cfg["input_size"],
                                        mean=rexnet_150_model.default_cfg.get("mean", IMAGENET_DEFAULT_MEAN),
                                        std=rexnet_150_model.default_cfg.get("std", IMAGENET_DEFAULT_STD),
                                        crop_pct=rexnet_150_model.default_cfg.get("crop_pct", 1.0),
                                        interpolation=rexnet_150_model.default_cfg.get("interpolation", "bilinear")
                                    )

# Load saved weights
rexnet_150_model.load_state_dict(torch.load(f="detection_plant_health_rexnet150.pth",
                                            map_location=torch.device("cpu"),  # load to CPU
                                           )
                                            )

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = rexnet_150_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    rexnet_150_model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(rexnet_150_model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Plant Disease Detection 🪴🍀🍁"
description = "A Rexnet_150 feature extractor computer vision model to classify images of plant with diseases."
article = "Created at [detection plant health application](https://github.com/JoachimLombardi/Portfolio/blob/master/Deep%20Learning/PyTorch/Classification/Computer%20vision/detection_plant_health_application.ipynb)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
