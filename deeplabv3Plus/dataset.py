import os
import cfg
import random
import torch
import PIL
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import torchvision.transforms.functional as Ft
# from torchvision.transforms.functional import InterpolationMode

import numpy as np
from PIL import Image
import scipy.io as scio

def image2label(image,colormap):
    cm2lbl = np.zeros(256**3)
    for i, cm in enumerate(colormap):
        # i means index
        # cm means categories
        # image's label for three channels pic is always in(R,G,B) integer mode
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
        # set the link between categories and integer
    image = np.array(image, dtype='int64')
    ix = ((image[:, :, 0]*256+image[:, :, 1])*256+image[:, :, 2])
    # calculate each pixel's integer
    image2 = cm2lbl[ix]
    # encode
    return image2

def label2image(prelabel,colormap):
    cm = np.array(colormap).astype('uint8')
    label = cm[prelabel]
    return label 


def inv_normalize_image(data):
    rgb_mean=np.array([0.485,0.456,0.406])
    rgb_std=np.array([0.229,0.224,0.225])
    data=data.astype('float32')*rgb_std+rgb_mean
    return data.clip(0,1)

def crop(data,label,high,width):
    # get the shape of the image
    im_width, im_high = data.size
    # randomly get the crop point in up_left corner
    if im_width == width:
        left = 0
    else:
        left = np.random.randint(0, im_width-width)
    if im_high == high:
        top = 0
    else:
        top = np.random.randint(0, im_high-high)
    # randomly get the crop point in down_right corner
    right = left + width
    bottom = top + high
    # crop the image
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    return data, label

def img_transforms(data, label, high, width,colormap):
    im_width, im_high = data.size
    random_ratio = random.uniform(0.5,2.0)
    target_high = int(random_ratio * im_high)
    target_width =  int(random_ratio * im_width)
    #target_size = max(im_width, im_high)
    data = Ft.resize(data, [target_high, target_width], PIL.Image.BILINEAR)
    label = Ft.resize(label,[target_high, target_width], PIL.Image.NEAREST) 
    n_width, n_high = data.size
    if n_width < width:
        diff = width - n_width
        data = Ft.pad(data, [diff//2, 0, diff-diff//2, 0])
        label = Ft.pad(label, [diff//2, 0, diff-diff//2, 0])
    if n_high < high:
        diff = high - n_high
        data = Ft.pad(data, [0, diff//2, 0, diff-diff//2])
        label = Ft.pad(label, [0, diff//2, 0, diff-diff//2])
    # im_width, im_high = data.size
    if n_width > width or n_high > high:
        data, label = crop(data, label, high, width)
      
    random_number = random.random()
    
 
    if random_number < 0.5:
        data = Ft.hflip(data)
        label = Ft.hflip(label)
    if random_number < 0.5:
        data = Ft.vflip(data)
        label = Ft.vflip(label)
         


    data_tfs = transforms.Compose([
        transforms.ColorJitter(brightness=(0.8,1.2), saturation=0.2, hue=(-0.2,0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    data = data_tfs(data)
    #label = Ft.resize(label, [high, width], interpolation=InterpolationMode.NEAREST)
    label = torch.from_numpy(image2label(label, colormap))
    return data, label

def val_img_transforms(data, label, high, width, colormap):
    im_width, im_high = data.size
    assert high == width
    data = Ft.resize(data, high, PIL.Image.BILINEAR)
    label = Ft.resize(label, high, PIL.Image.NEAREST)
    data = Ft.center_crop(data, high)
    label = Ft.center_crop(label, high)
    data_tfs = transforms.Compose([
        # transforms.Resize((high, width), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    data = data_tfs(data)
    #label = Ft.resize(label, [high, width], interpolation=InterpolationMode.NEAREST)
    label = torch.from_numpy(image2label(label, colormap))
    return data, label

def read_image_path(dataroot='traindata', labelroot='trainlabel', train_list = 'where is train.txt'):
    image = np.loadtxt(train_list, dtype=str)
    n = len(image)
    data, label = [None]*n, [None]*n
    for i, fname in enumerate(image):
        data[i] = os.path.join(dataroot, ('%s.jpg'%(fname)))
        label[i] = os.path.join(labelroot, ('%s.png'%(fname)))          
    return data, label

def read_val_image_path(dataroot='valdata', labelroot='vallabel', val_list = 'where is val.txt'):
    image = np.loadtxt(val_list, dtype=str)
    n = len(image)
    data, label = [None]*n, [None]*n
    for i, fname in enumerate(image):
        data[i] = os.path.join(dataroot, ('%s.jpg'%(fname)))
        label[i] = os.path.join(labelroot, ('%s.png'%(fname)))
    return data, label

# train dataset
class MyDataset(Data.Dataset):
    def __init__(self, data_root, label_root, train_list, high, width, imgtransform, colormap):
        self.data_root = data_root
        self.label_root = label_root
        self.high = high
        self.width = width
        self.imgtransform = imgtransform
        self.colormap = colormap
        self.data_list, self.label_list = read_image_path(data_root, label_root, train_list)

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        image = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.imgtransform(image, label, self.high, self.width, self.colormap)
        return img, label

    def __len__(self):
        return len(self.data_list)