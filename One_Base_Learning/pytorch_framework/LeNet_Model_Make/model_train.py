from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from lenet_model_setup import LeNet


def train_val_data_process_func():
    train_data = FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]
        ),
        download=True,
    )


if __name__ == "__main__":
    pass
