import torch
import torchvision
from torch import nn

def create_vitb16_model(num_classes: int = 3,
                        seed: int = 42) -> tuple[nn.Module, torchvision.transforms.Compose]:
    """
    Create a ViT-B/16 feature extractor model and transforms

    Args:
        num_classes (int, optional): number of classes. Defaults to 3.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        model (nn.Module): ViT-B/16 feature extractor model, 
        transforms (torchvision.transforms.Compose): ViT-B/16 images transforms
    """
    # Get the model weights
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    # Get automatic transforms from pretrained ViT weights
    transforms = weights.transforms()
    # Get the model architecture with pretrained weights
    model = torchvision.models.vit_b_16(weights=weights)
    # Get the last layer input
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            in_features = module.in_features     
    # Get its head
    torch.manual_seed(seed)
    model.heads = nn.Linear(in_features=in_features, out_features=num_classes)
    return model, transforms
