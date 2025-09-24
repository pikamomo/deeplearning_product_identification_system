__all__ = ["ImageDataSet"]

import os
import re

from PIL import Image

import torch
from torch.utils.data import Dataset
from similarity_config import *

def sorted_alphanum(img_names):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name: [convert(x) for x in re.split(r'([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)

class ImageDataSet(Dataset):
    def __init__(self, image_dir, transform=None):
        super(ImageDataSet, self).__init__()
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(self.main_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        image = Image.open(image_loc).convert('RGB')
        if self.transform is not None:
            tenser_img = self.transform(image)
        else:
            raise ValueError("transform is not defined")

        return tenser_img, tenser_img

if __name__ == "__main__":
    # image_names = os.listdir("../common/dataset")
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WEDTH)),
        transforms.ToTensor()
    ])
    dataset = ImageDataSet(IMG_PATH, transform=transform)
    print(len(dataset))