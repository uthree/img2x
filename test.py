from model import *
from dataset import *
import torch
from PIL import Image

GAN = SRGAN()
if os.path.exists("model.pt"):
    GAN.load_state_dict(torch.load("model.pt"))
    print("Loaded Model")
G = GAN.generator

img = Image.open('./demo_img.png')
W, H = img.size
out = G.resize(img, (W*4, H*4))
out.save('./demo_img_out.png')
