import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn as nn

# For FID and IS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Import your Generator class and options
# from implementations.dcgan.dcgan_modified import Generator

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/dcgan_modified_128_image256_400.pth"
TENSOR_DIR = "wikiart_batched_128_tensors_256"
N_SAMPLES = 1000
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
IMAGE_SIZE = 256

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = IMAGE_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# --- LOAD GENERATOR ---
generator = Generator().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# --- GENERATE FAKE IMAGES ---
fake_images = []
with torch.no_grad():
    for _ in tqdm(range(0, N_SAMPLES, BATCH_SIZE), desc="Generating fake images"):
        z = torch.randn(min(BATCH_SIZE, N_SAMPLES - len(fake_images)), LATENT_DIM, device=DEVICE)
        gen_imgs = generator(z)
        fake_images.append(gen_imgs)

fake_images = torch.cat(fake_images, dim=0)[:N_SAMPLES]

class BatchTensorDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.batch_files = sorted(os.listdir(tensor_dir))  # Sort to maintain order

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        batch_path = os.path.join(self.tensor_dir, self.batch_files[idx])
        batch = torch.load(batch_path)  # Load the entire batch
        return batch

real_dataset = BatchTensorDataset(TENSOR_DIR)
real_loader = DataLoader(real_dataset, batch_size=1, shuffle=True, num_workers=4)
real_images = []
for batch in tqdm(real_loader, desc="Loading real images"):
    real_images.append(batch[0].to(DEVICE))
    if len(real_images) * 128 >= N_SAMPLES:
        break
real_images = torch.cat(real_images, dim=0)[:N_SAMPLES]

# --- FID & IS COMPUTATION ---
# Images must be in [0, 1] and shape (N, 3, H, W)
def preprocess(imgs):
    # If images are in [-1, 1], convert to [0, 1]
    imgs = (imgs + 1) / 2 if imgs.min() < 0 else imgs
    imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
    return imgs

fake_images = preprocess(fake_images)
real_images = preprocess(real_images)

# FID
fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
for i in range(0, N_SAMPLES, BATCH_SIZE):
    fid.update(real_images[i:i+BATCH_SIZE].to(DEVICE), real=True)
    fid.update(fake_images[i:i+BATCH_SIZE].to(DEVICE), real=False)
fid_score = fid.compute().item()

# Inception Score (on fake images)
is_metric = InceptionScore(splits=10).to(DEVICE)
for i in range(0, N_SAMPLES, BATCH_SIZE):
    is_metric.update(fake_images[i:i+BATCH_SIZE].to(DEVICE))
is_mean, is_std = is_metric.compute()
is_mean, is_std = is_mean.item(), is_std.item()

print(f"FID: {fid_score:.4f}")
print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")