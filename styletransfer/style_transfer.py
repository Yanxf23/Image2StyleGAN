import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append("../")
from stylegan_layers import G_mapping, G_synthesis

# Device setup
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_generator():
    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('g_synthesis', G_synthesis(resolution=1024))
    ]))
    g_all.load_state_dict(torch.load('../weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt', map_location=device))
    g_all.eval().to(device)
    return g_all[0], g_all[1]

def load_latents(content_path, style_path):
    content_latent = np.load(content_path, allow_pickle=True)
    style_latent = np.load(style_path, allow_pickle=True)
    return torch.from_numpy(content_latent).to(device), torch.from_numpy(style_latent).to(device)

def style_transfer(content_latent, style_latent, content_layers):
    combined_latent = content_latent.clone()
    combined_latent[:, content_layers:, :] = style_latent[:, content_layers:, :]
    return combined_latent

def generate_image(g_synthesis, latent_code):
    img = g_synthesis(latent_code)
    return (img + 1.0) / 2.0  # Normalize to [0,1]

def tensor_to_numpy(img_tensor):
    return img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)

def run_style_transfer(content_latent_path, style_latent_path):
    g_mapping, g_synthesis = load_generator()
    content_latent, style_latent = load_latents(content_latent_path, style_latent_path)

    fig, axes = plt.subplots(1, 7, figsize=(21, 5))  # 7 steps (1, 4, 7, 10, 13, 16, 18)
    content_layers_list = list(range(0, 19, 3))  # Steps of 3

    for i, content_layers in enumerate(content_layers_list):
        combined_latent = style_transfer(content_latent, style_latent, content_layers)
        combined_img = generate_image(g_synthesis, combined_latent)
        axes[i].imshow(tensor_to_numpy(combined_img))
        axes[i].set_title(f"Content Layers: {content_layers}")
        axes[i].axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig("style_transfer3.png")
    plt.close()

content_latent_path = r"C:\Users\mobil\Desktop\25spring\stylePalm\Image2StyleGAN\styletransfer\palm_example\0009.npy"
style_latent_path = r"C:\Users\mobil\Desktop\25spring\stylePalm\Image2StyleGAN\styletransfer\palm_example\0036.npy"

run_style_transfer(content_latent_path, style_latent_path)
