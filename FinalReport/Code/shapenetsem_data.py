from os import listdir
from os.path import isfile, join
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ShapeNetSem(Dataset):
    def __init__(self, data_dir, onlyfiles_base, metadata_filtered_dict, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((128 , 128)),
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        self.items = []

        for basefilename in onlyfiles_base:
            file_paths = glob.glob(f"{self.data_dir}/**/{basefilename}.png", recursive=True)
            for file_path in file_paths:
                if basefilename not in metadata_filtered_dict:
                    continue
                label = metadata_filtered_dict[basefilename]
                self.items.append((file_path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long).squeeze()
        return image, label
