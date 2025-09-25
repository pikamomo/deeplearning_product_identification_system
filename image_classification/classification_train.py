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
from classification_config import *
from classification_engine import train_step, test_step
from image_classification.classification_data import ImageLabelDataSet
from image_classification.classification_model import Classifier
from image_denoising.denoising_model import ConvDenoiser

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    utils.seed_everything(SEED)

    transform = T.Compose([
     T.Resize((IMG_HEIGHT, IMG_WiDTH)),
     T.ToTensor()
    ])

    print("create dataset")
    dataset = ImageLabelDataSet(IMG_PATH, FASHION_LABELS_PATH,transform=transform)
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
    classifier = Classifier()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters())
    classifier.to(device)

    min_test_loss = 9999

    print("start training")
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(classifier, train_loader, loss, optimizer, device)
        print("epoch: {}, train_loss: {}".format(epoch, train_loss))

        test_loss, test_correct_num = test_step(classifier, test_loader, loss, device)
        accuracy = test_correct_num / len(test_dataset)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(classifier.state_dict(), CLASSIFIER_MODEL_NAME)

    print("finish training")