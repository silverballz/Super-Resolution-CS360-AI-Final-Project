# ============================================
# ESRGAN for Artwork Super Resolution
# Author: Minimal Training Script
# ============================================

import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import vgg19
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Dataset
# ============================================

class ArtworkDataset(Dataset):
    def __init__(self, hr_dir, scale=4):
        self.hr_images = glob.glob(hr_dir + "/*.png")
        self.scale = scale

        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):

        img = Image.open(self.hr_images[idx]).convert("RGB")
        hr = self.hr_transform(img)

        lr = transforms.Resize(
            hr.shape[1]//self.scale,
            interpolation=Image.BICUBIC
        )(transforms.ToPILImage()(hr))

        lr = transforms.Resize(
            hr.shape[1],
            interpolation=Image.BICUBIC
        )(lr)

        lr = transforms.ToTensor()(lr)

        return lr, hr


# ============================================
# RRDB Block
# ============================================

class DenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf+gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):

        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x,x1],1)))
        x3 = self.lrelu(self.conv3(torch.cat([x,x1,x2],1)))
        x4 = self.lrelu(self.conv4(torch.cat([x,x1,x2,x3],1)))
        x5 = self.conv5(torch.cat([x,x1,x2,x3,x4],1))

        return x5*0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf):
        super().__init__()

        self.db1 = DenseBlock(nf)
        self.db2 = DenseBlock(nf)
        self.db3 = DenseBlock(nf)

    def forward(self,x):
        return self.db3(self.db2(self.db1(x))) * 0.2 + x


# ============================================
# Generator
# ============================================

class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23):
        super().__init__()

        self.conv_first = nn.Conv2d(3, nf, 3,1,1)

        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])

        self.conv_body = nn.Conv2d(nf,nf,3,1,1)

        self.up1 = nn.Conv2d(nf,nf,3,1,1)
        self.up2 = nn.Conv2d(nf,nf,3,1,1)

        self.conv_hr = nn.Conv2d(nf,nf,3,1,1)
        self.conv_last = nn.Conv2d(nf,3,3,1,1)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):

        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.up1(
                nn.functional.interpolate(fea, scale_factor=2)
            )
        )

        fea = self.lrelu(
            self.up2(
                nn.functional.interpolate(fea, scale_factor=2)
            )
        )

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return out


# ============================================
# Discriminator
# ============================================

class Discriminator(nn.Module):
    def __init__(self):

        super().__init__()

        def block(in_c,out_c,stride):

            return nn.Sequential(
                nn.Conv2d(in_c,out_c,3,stride,1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        self.net = nn.Sequential(

            block(3,64,1),
            block(64,64,2),

            block(64,128,1),
            block(128,128,2),

            block(128,256,1),
            block(256,256,2),

            block(256,512,1),
            block(512,512,2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,100),
            nn.LeakyReLU(0.2),
            nn.Linear(100,1)
        )

    def forward(self,x):
        return self.net(x)


# ============================================
# VGG Perceptual Loss
# ============================================

class VGGFeatureExtractor(nn.Module):
    def __init__(self):

        super().__init__()

        vgg = vgg19(pretrained=True).features[:35]
        self.features = vgg.eval()

        for p in self.features.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.features(x)


# ============================================
# Training Setup
# ============================================

dataset = ArtworkDataset("data/artwork_hr")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

G = RRDBNet().to(device)
D = Discriminator().to(device)

vgg = VGGFeatureExtractor().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()

opt_g = optim.Adam(G.parameters(), lr=1e-4)
opt_d = optim.Adam(D.parameters(), lr=1e-4)

# ============================================
# Training Loop
# ============================================

epochs = 100

for epoch in range(epochs):

    for i,(lr,hr) in enumerate(loader):

        lr = lr.to(device)
        hr = hr.to(device)

        valid = torch.ones(hr.size(0),1).to(device)
        fake = torch.zeros(hr.size(0),1).to(device)

        # -----------------
        # Train Generator
        # -----------------

        sr = G(lr)

        pred_fake = D(sr)

        loss_gan = criterion_gan(pred_fake,valid)

        loss_l1 = criterion_l1(sr,hr)

        loss_perc = criterion_l1(
            vgg(sr),
            vgg(hr)
        )

        loss_g = loss_l1 + 0.01*loss_gan + 0.1*loss_perc

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # -----------------
        # Train Discriminator
        # -----------------

        pred_real = D(hr)
        loss_real = criterion_gan(pred_real,valid)

        pred_fake = D(sr.detach())
        loss_fake = criterion_gan(pred_fake,fake)

        loss_d = (loss_real + loss_fake)/2

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        if i % 50 == 0:

            print(
                f"Epoch {epoch} Batch {i}",
                f"G Loss {loss_g.item():.4f}",
                f"D Loss {loss_d.item():.4f}"
            )

    # Save sample
    vutils.save_image(sr,"sample_sr.png")

# ============================================
# Save model
# ============================================

torch.save(G.state_dict(),"esrgan_artwork.pth")