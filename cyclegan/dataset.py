import os
import os.path as osp
from torch.utils.data import Dataset, random_split
from torchvision import transforms, io
from PIL import Image
import numpy as np
import config

class PhotoMonetDataset(Dataset):
    def __init__(self, root_photo, root_monet, transform=None):
        self.root_photo = root_photo
        self.root_monet = root_monet
        self.transform = transform
        if config.DEBUG:
            self.monet_images = os.listdir(self.root_monet)[:200]
            self.photo_images = os.listdir(self.root_photo)[:200]
        else:
            self.monet_images = os.listdir(self.root_monet)
            self.photo_images = os.listdir(self.root_photo)
        self.num_monet = len(self.monet_images)
        self.num_photo = len(self.photo_images)
        self.len_dataset = max(self.num_monet, self.num_photo)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        monet_img = self.monet_images[index%self.num_monet]
        photo_img = self.photo_images[index%self.num_photo]

        monet_img = osp.join(self.root_monet, monet_img)
        photo_img = osp.join(self.root_photo, photo_img)

        monet_img = np.array(Image.open(monet_img).convert("RGB"))
        photo_img = np.array(Image.open(photo_img).convert("RGB"))

        if self.transform:
            monet_img = self.transform(monet_img)
            photo_img = self.transform(photo_img)

        return monet_img, photo_img


if __name__ == "__main__":
    PROJECT_ROOT = "../gan-getting-started"
    dataset = PhotoMonetDataset(osp.join(PROJECT_ROOT, "monet_jpg"), 
                                osp.join(PROJECT_ROOT, "photo_jpg"))
    
    idx = 100
    print(dataset.__getitem__(100))