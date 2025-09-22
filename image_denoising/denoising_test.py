import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from common import utils
from denoising_config import *
from denoising_engine import train_step, test_step
from image_denoising.denoising_data import ImageDataSet
from image_denoising.denoising_model import ConvDenoiser

def test(denoiser, test_loader, device):
    denoiser.eval()
    data_iter = iter(test_loader)
    noise_images, original_images = next(data_iter)
    denoiser = denoiser.to(device)
    noise_images = noise_images.to(device)
    output = denoiser(noise_images)

    noise_img = noise_images.permute(0, 2, 3, 1).cpu().numpy()
    output_imgs = output.permute(0, 2, 3, 1).detach().cpu().numpy()

    original_imgs = original_images.permute(0, 2, 3, 1).detach().cpu().numpy()

    fig, axes = plt.subplots(3, 10, figsize=(25, 4))
    for imgs, row in zip([noise_img, output_imgs, original_imgs], axes):
        for img, ax in zip(imgs, row):
            ax.imshow(img)
            ax.axis('off')
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    utils.seed_everything(SEED)

    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WEDTH)),
        T.ToTensor()
    ])

    print("create dataset")
    dataset = ImageDataSet(IMG_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    print("create loader")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    print("create model")
    loaded_denoiser = ConvDenoiser()
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)

    test(loaded_denoiser, test_loader, device)