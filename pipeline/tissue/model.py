"""
Load Subham's tissue classifier.

It's a ResNet18 with the final classification head replaced to output
3 classes: granulation, necrosis, slough.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from pipeline.config import TISSUE_MODEL_PATH, TISSUE_CLASSES


def load_tissue_model(device=None):
    """
    Load the trained ResNet18 tissue classifier.

    Args:
        device: torch device, or None to auto-pick.

    Returns:
        model:  ResNet18 with 3-class head, in eval mode, on device
        device: device we ended up using
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the architecture (same as in training) — weights=None because we'll
    # load our own state dict right after.
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(TISSUE_CLASSES))   # swap head to 3 classes

    # Load the trained weights from disk
    state_dict = torch.load(str(TISSUE_MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()   # turn off dropout / batchnorm updates for inference

    print(f"[tissue] ResNet18 model loaded on {device}")
    return model, device
