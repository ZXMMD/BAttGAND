from __future__ import print_function
import shutil
import torch.nn
import os
import cv2
import glob
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
from model import ConvolutionalAutoencoder



def image_loader(image_name, input_shape):
    loader = transforms.Compose([transforms.Scale(input_shape), transforms.ToTensor()])

    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def recon_image(model, image_name, input_shape):
    image = image_loader(image_name, input_shape)
    image_recon = model(image.clone().detach())

    image_recon_cv = image_recon[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image_recon_cv = cv2.cvtColor(image_recon_cv, cv2.COLOR_RGB2BGR)

    return image_recon_cv


# original_image_path = './GAN_dataset/StyleGAN'
# transformed_image_path = './StyleGAN_ad'
def gan_fingerprint_removal(original_image_path, transformed_image_path, model_dir):
    image_list = glob.glob(original_image_path + '/*.*')
    if not os.path.exists(transformed_image_path):
        os.mkdir(transformed_image_path)

    model = ConvolutionalAutoencoder(latent_code_size=32)
    model.load_state_dict(torch.load(model_dir))

    for image_path in tqdm(image_list):
        output_image_path = image_path.replace(original_image_path, transformed_image_path)
        img = recon_image(model, image_path, 224)
        cv2.imwrite(output_image_path, img)

if __name__ == "__main__":
    original_image_path = './GAN_dataset/StyleGAN'
    transformed_image_path = './StyleGAN_ad'
    model_dir = './weights/BAttGAND.pth'
    gan_fingerprint_removal(original_image_path, transformed_image_path, model_dir)


