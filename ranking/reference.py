import os
import torch
import clip
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from PIL import Image
from urllib import request
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
import pdb 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

preprocess = T.Compose([
   T.Resize(224),
   T.CenterCrop(224),
   T.ToTensor(),
   T.Normalize(
       mean=[0.48145466, 0.4578275, 0.40821073],
       std=[0.26862954, 0.26130258, 0.27577711]
   )
])



text_tokens = clip.tokenize(["rabbit", "rabit", "rbbit"]).cuda(device=device)
id = np.zeros(5)

files = os.listdir("rabbit")
similarity_list = []

with open("test.txt", "w") as file_write:
    for file in files:
        file_open = os.path.join("rabbit", file)
        image = Image.open(file_open).convert("RGB")
        image = preprocess(image)
        image_input = image.cuda(device=device)

        with torch.no_grad():
            image_input= image_input.unsqueeze(0)
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        max_val = similarity.argmax(axis=0)
        id[max_val]+=1
        to_write = str(similarity.flatten()) + " : " + str(max_val)
        file_write.write(to_write)
        file_write.write("\n")
        
with open("test.txt", "a") as file_write:
    file_write.write(str(id))