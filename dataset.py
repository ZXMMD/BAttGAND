import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
import torchvision.transforms.functional as F


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG'])


# Paired Image Datasets
class PairedImgDataset(Dataset):
    def __init__(self, root, tag='train', img_size=224) -> None:
        super().__init__()
        self.img_size = img_size  
        self.dir = os.path.join(root, tag)  

        self.in_files = sorted(os.listdir(os.path.join(self.dir, 'GAN')))  
        self.ad_files = sorted(
            os.listdir(os.path.join(self.dir, 'Adversarial')))  

        assert len(self.in_files) == len(self.ad_files), "GAN numbers doesn't match Adversarial number."
        self.data_size = len(self.in_files)

    def __getitem__(self, index):
        in_nm, ad_nm = self.in_files[index], self.ad_files[index]

        in_img = Image.open(os.path.join(self.dir, 'GAN', in_nm))
        ad_img = Image.open(os.path.join(self.dir, 'Adversarial', ad_nm))

        resize = TF.Resize(224)
        totensor = TF.ToTensor()

        in_img = resize(in_img) 
        in_img = totensor(in_img) 

        ad_img = totensor(ad_img)  

        return in_img, ad_img

    def __len__(self):
        return self.data_size


# define train loader
def get_train_loader(data_dir, batch_size, num_workers, shuffle=True, img_size=224):
    train_dataset = PairedImgDataset(data_dir, tag='train', img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)
    return train_loader, len(train_dataset)



