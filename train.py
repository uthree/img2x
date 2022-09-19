from model import *
from dataset import *
import torch

GAN = SRGAN()
MAX_DATASET_LEN=1000
NUM_EPOCH=3
BATCH_SIZE=1

if os.path.exists("model.pt"):
    GAN.load_state_dict(torch.load("model.pt"))
    print("Loaded Model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GAN.to(device)

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_256x/"], max_len=MAX_DATASET_LEN)
ds.set_size(128)

GAN.train(ds, batch_size=BATCH_SIZE, train_discriminator=False, num_epoch=NUM_EPOCH, apply_vgg_loss=False)
torch.save(GAN.state_dict(), "./model.pt")


