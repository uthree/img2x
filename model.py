import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import math
import numpy as np

from tqdm import tqdm

class SRAutoPadder(nn.Module):
    def __init__(self, model, unit_pixel=16):
        super().__init__()
        self.model = model
        self.unit_pixel = unit_pixel

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        if x.shape[2] % self.unit_pixel != 0:
            ha = self.unit_pixel - (x.shape[2] % self.unit_pixel)
            x = torch.cat([x, x[:, :, -ha:]], dim=2)
        if x.shape[3] % self.unit_pixel != 0:
            wa = self.unit_pixel - (x.shape[3] % self.unit_pixel)
            x = torch.cat([x, x[:, :, :, -wa:]], dim=3)
        x = self.model(x)[:, :, :H*2, :W*2]
        return x

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x): # x: [N, C, H, W]
        m = x.mean(dim=1, keepdim=True)
        s = ((x - m) ** 2).mean(dim=1, keepdim=True)
        x = (x - m) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dim_ffn=None, kernel_size=7, norm=True):
        super(ConvNeXtBlock, self).__init__()
        if dim_ffn == None:
            dim_ffn = channels * 4
        self.c1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, padding_mode='replicate', groups=channels)
        self.norm = ChannelNorm(channels) if norm else nn.Identity()
        self.c2 = nn.Conv2d(channels, dim_ffn, 1, 1, 0)
        self.act = nn.LeakyReLU(0.2)
        self.c3 = nn.Conv2d(dim_ffn, channels, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.act(x)
        x = self.c3(x)
        return x + res

# Input: [N, input_channels, H, W]
# Output: [N, output_features]
class ConvNeXt(nn.Module):
    def __init__(self, input_channels=3, stages=[3, 3, 3, 3], channels=[32, 64, 128, 256], output_features=256, minibatch_std=False, stem=False):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, channels[0], 4, 4, 0) if stem else nn.Conv2d(input_channels, channels[0], 3, 1, 1)
        seq = []
        if minibatch_std:
            self.out_linear = nn.Sequential(nn.Linear(channels[-1]+1, output_features), nn.Linear(output_features, output_features))
        else:
            self.out_linear = nn.Linear(channels[-1], output_features)
        self.mb_std = minibatch_std
        for i, (l, c) in enumerate(zip(stages, channels)):
            for _ in range(l):
                seq.append(ConvNeXtBlock(c))
            if i != len(stages)-1:
                seq.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
                seq.append(ChannelNorm(channels[i+1]))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.stem(x)
        x = self.seq(x)
        x = torch.mean(x,dim=[2,3], keepdim=False)
        if self.mb_std:
            mb_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1)
            x = torch.cat([x, mb_std], dim=1)
            x = self.out_linear(x)
        else:
            x = self.out_linear(x)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

# SRNet
class SRNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, stages=[3, 3, 9, 3], channels=[32, 64, 128, 256], tanh=True, kernel_size=7, ffn_mult=4, last_channels=8):
        super().__init__()
        self.encoder_first = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.decoder_last = nn.Sequential(
                nn.Conv2d(channels[0], last_channels, 1, 1, 0),
                nn.Upsample(scale_factor=2),
                ConvNeXtBlock(last_channels),
                nn.Conv2d(last_channels, output_channels, 1, 1, 0))
        self.tanh = nn.Tanh() if tanh else nn.Identity()
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = nn.Sequential(*[ConvNeXtBlock(c, kernel_size=kernel_size, dim_ffn=c*ffn_mult) for _ in range(l)])
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0), ChannelNorm(channels[i+1]))
            dec_stage = nn.Sequential(*[ConvNeXtBlock(c, kernel_size=kernel_size, dim_ffn=c*ffn_mult, norm=False) for _ in range(l)])
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Upsample(scale_factor=2),nn.Conv2d(channels[i+1], channels[i], 1, 1, 0))
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x):
        x = self.encoder_first(x)
        skips = []
        for l in self.encoder_stages:
            x = l.stage(x)
            skips.insert(0, x)
            x = l.ch_conv(x)
        for i, (l, s) in enumerate(zip(self.decoder_stages, skips)):
            x = l.ch_conv(x)
            x = l.stage(x + s)
        x = self.decoder_last(x)
        x = self.tanh(x)
        return x

    def resize(self, image: Image, size):
        # convert to tensor
        device = self.parameters().__next__().device
        h, w = image.size
        hr, wr = size
        hs, ws = hr/h, wr/w
        scale_factor = max(hs, ws)
        img = torch.from_numpy(np.transpose(np.array(image).astype(np.float32) / 127.5 - 1.0, (2, 0, 1))).to(device)
        img = torch.unsqueeze(img, 0)
        resize_count = math.ceil(math.log2(scale_factor))
        model = SRAutoPadder(self)
        with torch.no_grad():
            for _ in range(resize_count):
                img = model.forward(img)
        img = img.detach()
        img = Image.fromarray((img[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB').resize(size)
        return img

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).features.eval()
        self.MSE = nn.MSELoss()

    def forward(self, x1, x2):
        return self.MSE(self.vgg(x1), self.vgg(x2)) + self.MSE(x1, x2)

class SRGAN(nn.Module):
    def __init__(self, g_channels=[32, 64, 128], g_stages=[2, 2, 2], d_channels=[32, 64, 128], d_stages=[2, 2, 2]):
        super().__init__()
        self.generator = SRNet(3, 3, g_stages, g_channels, tanh=True)
        self.discriminator = ConvNeXt(3, d_stages, d_channels, 1)

    def train(self, dataset, batch_size=1, num_epoch=1, train_discriminator=False, apply_vgg_loss=True):
        bar = tqdm(total=len(dataset) * num_epoch)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        G = self.generator
        D = self.discriminator
        opt_g = optim.RAdam(G.parameters())
        opt_d = optim.RAdam(D.parameters())
        device = self.parameters().__next__().device
        criterion_g = (VGGLoss() if apply_vgg_loss else nn.MSELoss()).to(device)
        MSE = nn.MSELoss()
        for epoch in range(num_epoch):
            for i, img in enumerate(dataloader):
                img = img.to(device)
                N = img.shape[0]
                img_in = F.avg_pool2d(img, kernel_size=2).detach()
                # Train Generator
                opt_g.zero_grad()
                fake = G(img_in)
                loss_g = criterion_g(fake, img)
                if train_discriminator:
                    loss_g += MSE(D(fake), (torch.zeros(N, 1, device=device)))
                loss_g.backward()
                opt_g.step()
                if train_discriminator:
                    # Train Discriminator
                    opt_d.zero_grad()
                    fake = fake.detach()
                    logit_fake = D(fake)
                    logit_real = D(img)
                    loss_df = MSE(logit_fake, torch.ones(N, 1, device=device))
                    loss_dr = MSE(logit_real, torch.zeros(N, 1, device=device))
                    loss_d = loss_df + loss_dr
                    loss_d.backward()
                    opt_d.step()

                    bar.set_description(desc=f"G.Loss:{loss_g.item():.4f} D.Loss: {loss_d.item():.4f}")
                else:
                    bar.set_description(desc=f"G.Loss:{loss_g.item():.4f}")
                bar.update(N)


