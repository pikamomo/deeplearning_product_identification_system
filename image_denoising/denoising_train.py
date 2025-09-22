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
    denoiser = ConvDenoiser()
    loss = nn.MSELoss()
    optimizer = optim.Adam(denoiser.parameters())
    denoiser.to(device)

    min_test_loss = 9999

    print("start training")
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(denoiser, train_loader, loss, optimizer, device)
        print("epoch: {}, train_loss: {}".format(epoch, train_loss))

        test_loss = test_step(denoiser, test_loader, loss, device)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)

    print("finish training")