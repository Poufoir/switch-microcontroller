import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
from functools import cache


class RegressionModel(nn.Module):
    def __init__(
        self, in_channels: int = 3, image_height: int = 640, image_width: int = 480
    ):
        self.image_height = image_height
        self.image_width = image_width
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (image_height // 4) * (image_width // 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.double()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (self.image_height // 4) * (self.image_width // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@cache
def get_model():
    model = RegressionModel()
    model.load_state_dict(torch.load("model_train.pt"))
    model.eval()
    return model


def predict_image(image: np.ndarray):
    model = get_model()
    frame = torch.from_numpy(image / 255).permute(2, 0, 1).type(torch.DoubleTensor)
    return torch.round(model(frame)[0]).type(torch.int)
