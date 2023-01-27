import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch 
import random

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]


class PersonalizedBase(Dataset):
    def __init__(self, size):
        
        self.size = size
        self.dir_base = "data/cup-data/template"
        self.dir_actu = "data/cup-data/one"

        self.all_files = []
        self.file_main = ""

        self.load_base()

        self.num_images = len(self.all_files)

    def load_base(self):

        all_files = os.listdir(self.dir_base)
        for file in all_files:
            self.all_files.append(os.path.join(self.dir_base, file))

        myfile = os.listdir(self.dir_actu)
        for file in myfile: 
            self.file_main = os.path.join(self.dir_actu,file)


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, i):
        example = {}

        text = random.choice(imagenet_templates_small).format("*")
        
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        image_style = Image.open(self.all_files[i % self.num_images])
        image_style = image_style.convert("RGB")
        image_style = image_style.resize((self.size, self.size), PIL.Image.Resampling.BILINEAR)
        image_style = transform(image_style).type(torch.FloatTensor)

        image_base = Image.open(self.file_main)
        image_base = image_base.convert("RGB")
        image_base = image_base.resize((self.size, self.size), PIL.Image.Resampling.BILINEAR)
        image_base = transform(image_base).type(torch.FloatTensor)

        example["text"] = text
        example["img_style"] = image_style
        example["img_base"] = image_base
        
        return example