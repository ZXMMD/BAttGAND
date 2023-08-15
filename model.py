import torch
from PIL import ImageFile
class ConvolutionalAutoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 32, 3, padding=1, dilation=2)

        self.trans1 = torch.nn.ConvTranspose2d(32, 128, 3, padding=1, dilation=2)
        self.trans2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(32, 3, 3, padding=1)
        self.mp = torch.nn.MaxPool2d(2, return_indices=True)
        self.up = torch.nn.MaxUnpool2d(2)
        self.relu = torch.nn.ReLU()

    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # [?, 32, 224, 224]
        s1 = x.size()
        x, ind1 = self.mp(x)  # [?, 32, 112, 112]
        x = self.conv2(x)
        x = self.relu(x)  # [?, 64, 112, 112]
        s2 = x.size()
        x, ind2 = self.mp(x)  # [?, 64, 56, 56]
        x = self.conv3(x)
        x = self.relu(x)  # [?, 128, 56, 56]
        s3 = x.size()
        x, ind3 = self.mp(x)  # [?, 128, 28, 28]
        x = self.conv4(x)
        x = self.relu(x)  # [?, 32, 28, 28]

        return x, ind1, s1, ind2, s2, ind3, s3

    def decoder(self, x, ind1, s1, ind2, s2, ind3, s3):
        x = self.trans1(x)
        x = self.relu(x)  # [128, 128, 28, 28]
        x = self.up(x, ind3, output_size=s3)  # [128, 128, 56, 56]
        x = self.trans2(x)
        x = self.relu(x)  # [128, 128, 56, 56]
        x = self.up(x, ind2, output_size=s2)  # [128, 128, 112, 112]
        x = self.trans3(x)
        x = self.relu(x)  # [128, 128, 112, 112]
        x = self.up(x, ind1, output_size=s1)  # [128, 128, 224, 224]
        x = self.trans4(x)
        x = self.relu(x)  # [128, 128, 224, 224]
        return x

    def forward(self, x):
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        output = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)
        return output
