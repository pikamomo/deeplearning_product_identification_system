__all__ = ["ImageLabelDataSet"]

import os
import re

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from classification_config import *


def sorted_alphanum(img_names):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name: [convert(x) for x in re.split(r'([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)

class ImageLabelDataSet(Dataset):
    def __init__(self, image_dir, label_path,transform=None):
        super(ImageLabelDataSet, self).__init__()
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(self.main_dir))
        self.labels = pd.read_csv(label_path)
        self.label_dict = dict(zip(self.labels['id'], self.labels['target']))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        image = Image.open(image_loc).convert('RGB')
        if self.transform is not None:
            tenser_img = self.transform(image)
        else:
            raise ValueError("transform is not defined")

        label = self.label_dict[idx]
        return tenser_img, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WiDTH)),
        transforms.ToTensor(),
    ])
    dataset = ImageLabelDataSet(IMG_PATH, FASHION_LABELS_PATH, transform)
    print(len(dataset))
