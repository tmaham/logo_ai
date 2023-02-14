from ldm.modules.encoders.modules import FrozenClipImageEmbedder as clipimg
from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder as cliptxt
import clip 
from PIL import Image
import os 
import pdb 
import torchvision.transforms as T
import torch
from font_list import font_list
import random
from PIL import Image, ImageOps
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np 
from torchvision.utils import save_image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from einops import rearrange, repeat
import csv
import math 
import sys



name = sys.argv[1]
letter = sys.argv[2]
mode = sys.argv[3]
color = sys.argv[4]


# letter = "C"
# name = "MUSIC"
# mode = "one"
# color = "color"


color_bg = [
    "white"
]

color_font = [
    "black"
]

one_line = [
    "logo of {}",
]

color_multi_font =[
    '#a3001b', 
    '#175c7a',
    '#5cd6ce',
    '#d1440c',
    '#1f1775',
    '#9e1e46',
    '#9b100d',
    '#d3760c',
    '#e8bb06',
    '#5135ad',
    '#366993',
    '#470e7c',
    '#070707',
    '#053d22',
    '#7a354c',
    '#7c0b03'
]

command = "rm -r output/*"
os.system(command)
make_black = True



main_dir = "/localhome/mta122/PycharmProjects/logo_ai/final_outputs/"
input_images_dir = os.path.join(main_dir, mode, color, name, letter, "samples")
output_ranked_dir = "output"
model_clip, preprocess = clip.load("ViT-L/14", device="cuda")


preprocess = T.Compose([
   T.Resize(224),
   T.CenterCrop(224),
   T.ToTensor(),
   T.Normalize(
       mean=[0.48145466, 0.4578275, 0.40821073],
       std=[0.26862954, 0.26130258, 0.27577711]
   )
])

def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textsize(txt, font)



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


text_tokens = clip.tokenize([name]).cuda(device="cuda")

all_images = os.listdir(input_images_dir)

config = OmegaConf.load("/localhome/mta122/PycharmProjects/logo_ai/configs/finetune/finetune.yaml")
ck = os.listdir("/localhome/mta122/PycharmProjects/logo_ai/logs")
li = os.path.join("/localhome/mta122/PycharmProjects/logo_ai/logs",ck[0],"checkpoints","last.ckpt")
model = load_model_from_config(config, li)  


list_img = []
list_clip_crit = []
list_disc_crit = []

for images in all_images:
    file_open = os.path.join(input_images_dir, images)
    image_org = Image.open(file_open).convert("RGB")
    image = preprocess(image_org)
    image_input = image.cuda(device="cuda")

    with torch.no_grad():
        image_input= image_input.unsqueeze(0)
        image_features = model_clip.encode_image(image_input).float()
        text_features = model_clip.encode_text(text_tokens).float()


    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    clip_crit = similarity.mean()
    # make the characters here 
    fontname = random.choice(font_list)
        
    if not make_black:
        colorText = random.choice(color_multi_font)
    else:
        colorText = random.choice(color_font)
    colorBackground = random.choice(color_bg)
    fontsize = 512   

    font = ImageFont.truetype(fontname, fontsize)
    width, height = getSize(letter, font)

    image = Image.new('RGB', (width+16, height+32+16), colorBackground)
    d = ImageDraw.Draw(image)
    d.text((8, 16), letter, fill=colorText, font=font)
    
    img = np.array(image).astype(np.uint8)  
    image_letter = Image.fromarray(img)
    image = preprocess(image_letter)

    image2 = image_input[0]
    image1 = image.cuda()
    image2_bw = (image2[0]+image2[1]+image2[2])/3
    image1_bw = (image1[0]+image1[1]+image1[2])/3
    
    # criteria = torch.nn.L1Loss()
    # loss = criteria(image2_bw,image1_bw)

    # name = str(loss.cpu().detach())+".jpg"
    # image_org.save("output/"+name)
    # name = str(loss.cpu().detach())+"_letter.jpg"
    # image_letter.save("output/"+name)
    # save_image(image2_bw, "output/"+str(loss.cpu().detach())+"image2.png")
    # save_image(image1_bw, "output/"+str(loss.cpu().detach())+"image1.png")
    
    
    image = image_org.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = rearrange(image, 'h w c -> c h w')
    image = torch.tensor(image).unsqueeze(dim=0).cuda()
    image = image.to(memory_format=torch.contiguous_format).float()
    image = model.encode_first_stage(image)
    image = model.get_first_stage_encoding(image).detach()

    output = model.discriminator(image, letter=0).view(-1)
    disc_crit = output[0].item()
    # save_image(image2, "output/"+str(output.cpu().detach())+"image.png")

    list_img.append(images)
    list_clip_crit.append(clip_crit)
    list_disc_crit.append(disc_crit)


#ANALYSIS HERE
list_clip_crit = (list_clip_crit-min(list_clip_crit))/(max(list_clip_crit)-min(list_clip_crit))
list_disc_crit = (np.array(list_disc_crit)-min(list_disc_crit))/(max(list_disc_crit)-min(list_disc_crit))
list_disc_crit = list_disc_crit * 100
list_clip_crit = list_clip_crit * 100
list_both_crit = list_clip_crit+list_disc_crit
list_mult_crit = list_clip_crit*list_disc_crit
list_total = list(zip(list_img,list_clip_crit,list_disc_crit,list_both_crit))


# "Clip Best"
# list_clip_crit.sort()
# print(list_clip_crit[0:4])
# "Disc Best"
# list_disc_crit.sort()
# print(list_disc_crit[0:4])
# "Both Best"
# list_both_crit.sort()
# print(list_both_crit[0:4])

i = 0
with open("analysis.csv", 'w') as fp:
    fp.write("Image"+ ","+ "Clip"+ ","+ "Disc"+","+"Add"+","+"Mult"+","+"Exp")
    fp.write("\n")
    while(i<len(list_total)):
        exp_val = math.exp(list_clip_crit[i]/20) + math.exp(list_disc_crit[i]/20)
        fp.write(list_img[i]+ ","+ str(list_clip_crit[i])+ ","+ str(list_disc_crit[i])+","+str(list_both_crit[i])+","+str(list_mult_crit[i])+","+str(exp_val))
        fp.write("\n")
        i+=1

main_dir = "/localhome/mta122/PycharmProjects/logo_ai/final_outputs/"
input_images_dir = os.path.join(main_dir, mode, color, name, letter)
command = "mv analysis.csv " +  input_images_dir
os.system(command)