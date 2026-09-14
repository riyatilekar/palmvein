import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18_64(nn.Module):
    def __init__(self):
        super(ResNet18_64, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = F.normalize(x, p=2, dim=1)
        return x
