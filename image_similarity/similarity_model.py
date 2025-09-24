__all__ = ['ConvEncoder','ConvDecoder']

import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.t_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.t_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.relu3 = nn.ReLU(inplace=True)

        self.t_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.t_conv5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.t_conv1(x)
        x = self.relu1(x)
        x = self.t_conv2(x)
        x = self.relu2(x)
        x = self.t_conv3(x)
        x = self.relu3(x)
        x = self.t_conv4(x)
        x = self.relu4(x)
        x = self.t_conv5(x)
        x = self.relu5(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    encoder = ConvEncoder()
    decoder = ConvDecoder()

    encoded_x = encoder(x)
    decoded_x = decoder(encoded_x)

    print(encoded_x.shape)
    print(decoded_x.shape)