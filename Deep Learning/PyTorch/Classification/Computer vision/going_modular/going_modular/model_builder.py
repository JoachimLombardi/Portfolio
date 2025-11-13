import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/

    Args:
        num_blocks: Number of convolutional layers
        input_shape: Number of channels in the input
        hidden_units: Number of hidden units
        output_shape: Number of channels in the output
    """
    def __init__(self, num_blocks: int, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        conv_blocks = []
        out_conv_blocks = 64
        for _ in range(num_blocks):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=input_shape,
                              out_channels=hidden_units,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=hidden_units,
                              out_channels=hidden_units,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            input_shape = hidden_units
            out_conv_blocks = out_conv_blocks // 2
        # Transform list of conv_blocks into a sequence of layers
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*out_conv_blocks*out_conv_blocks, # we divide by 2 for each conv_blocks
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_blocks(x)) # <- leverage the benefits of operator fusion

