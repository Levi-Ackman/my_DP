from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra=root_zebra
        self.root_horse=root_horse
        self.transform=transform

        self.zebra_images=os.listdir(root_zebra)
        self.horse_iamges=os.listdir(root_horse)
        self.len_dataset=max(len(self.zebra_images),len(self.horse_iamges))  # take the max_len of each img_num as teh real len of this dataset

        self.len_zebra=len(self.zebra_images)
        self.len_horse=len(self.horse_iamges)

    def __getitem__(self, index) :
        ## in order to make sure the img_pairs are coupled 
        zebra_img_filename=self.zebra_images[index % self.len_zebra]
        horse_img_filename=self.horse_iamges[index % self.len_horse]

        zebra_img_path=os.path.join(self.root_zebra, zebra_img_filename)
        horse_img_path=os.path.join(self.root_horse, horse_img_filename)

        zebra_img=Image.open(zebra_img_path)
        horse_img=Image.open(horse_img_path)

        if self.transform:
            zebra_img=self.transform['zebra'](zebra_img)
            horse_img=self.transform['horse'](horse_img)
        
        return zebra_img,horse_img

    def __len__(self):
        return self.len_dataset  ## return the maxium length