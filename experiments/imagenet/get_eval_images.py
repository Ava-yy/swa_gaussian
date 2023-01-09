import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def main():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    val_path = Path("../../imagenet/val")
    val_dataset = ImageFolder(val_path, transform=transform)

    for i, (img, label) in tqdm(enumerate(val_dataset)):
        img.save("../../imagenet/eval_images_dog/"+str(i)+".jpeg")


if __name__ == "__main__":
    main()