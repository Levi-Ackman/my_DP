import torch
from torchvision import transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Here we actually used oil_painting 2 photo data but the mechanism behind is the same
ZEBRA_TRAIN_DIR = "D:/320906183/Ads/kx/Gan/dataset/Zebra_Horse_Cycle_GAN/vangogh2photo/vangogh2photo/trainA"
HORSE_TRAIN_DIR = "D:/320906183/Ads/kx/Gan/dataset/Zebra_Horse_Cycle_GAN/vangogh2photo/vangogh2photo/trainB"

ZEBRA_VAL_DIR = "D:/320906183/Ads/kx/Gan/dataset/Zebra_Horse_Cycle_GAN/vangogh2photo/vangogh2photo/testA"
HORSE_VAL_DIR = "D:/320906183/Ads/kx/Gan/dataset/Zebra_Horse_Cycle_GAN/vangogh2photo/vangogh2photo/testB"

BATCH_SIZE = 1
IMG_SIZE=256
IMG_CHANNELS=3
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 8
NUM_EPOCHS = 3
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

## take a note, we  need transfer the img to tensor before we do normalization!

data_transform= transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5 for _ in range(IMG_CHANNELS)],std=[0.5 for _ in range(IMG_CHANNELS)]),
        ])
horse_zebra_transform={
    'horse':data_transform,
    'zebra':data_transform,
}