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
from similarity_config import *
from similarity_engine import train_step, test_step, create_embeading
from image_similarity.similarity_data import ImageDataSet
from similarity_model import ConvEncoder, ConvDecoder

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

    full_loader = DataLoader(dataset, batch_size=FULL_BATCH_SIZE)

    print("create model")
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    loss = nn.MSELoss()
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(autoencoder_params, lr=LEARNING_RATE)
    encoder.to(device)
    decoder.to(device)
    #
    # min_test_loss = 9999
    #
    # print("start training")
    # for epoch in tqdm(range(EPOCHS)):
    #     train_loss = train_step(encoder, decoder, train_loader, loss, optimizer, device)
    #     print("epoch: {}, train_loss: {}".format(epoch, train_loss))
    #
    #     test_loss = test_step(encoder, decoder, test_loader, loss, device)
    #
    #     if test_loss < min_test_loss:
    #         min_test_loss = test_loss
    #         torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
    #         torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
    # print("finish training")
    encoder_state_dict = torch.load(ENCODER_MODEL_NAME, map_location=device)
    encoder.load_state_dict(encoder_state_dict)

    embeddings = create_embeading(encoder, full_loader, device)
    vec_embeddings = embeddings.detach().cpu().numpy().reshape(embeddings.shape[0], -1)
    np.save(EMBEADDING_NAME, vec_embeddings)
    print(embeddings.shape)
    print(vec_embeddings.shape)