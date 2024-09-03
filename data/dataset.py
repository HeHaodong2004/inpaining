import sys
import torch.utils.data as data
from os import listdir
from utils.tools import default_loader, is_image_file, normalize
import os

import torchvision.transforms as transforms


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, random_crop=True):
        super(Dataset, self).__init__()
        self.ground_truth = [x for x in listdir(f"{data_path}/full") if is_image_file(x)]
        self.partial_map = [x for x in listdir(f"{data_path}/part") if is_image_file(x)]
        self.map_mask = [x for x in listdir(f"{data_path}/mask") if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop

    def crop_img(self, path, img_type=None):
        img = default_loader(path)
        if img_type == 'binary':
            img = img.convert('1')
        elif img_type == 'grayscale':
            img = img.convert('L')
        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)
        return img


    def __getitem__(self, index):
        partial_img = self.crop_img(os.path.join(f"{self.data_path}/part", self.partial_map[index]), img_type='grayscale')
        mask_img = self.crop_img(os.path.join(f"{self.data_path}/mask", self.map_mask[index]), img_type='binary')
        map_id = self.partial_map[index].split('_')[0]
        ground_truth = self.crop_img(os.path.join(f"{self.data_path}/full", f"{map_id}.png"), img_type='grayscale')
        return ground_truth, partial_img, mask_img

    def __len__(self):
        return len(self.partial_map)
