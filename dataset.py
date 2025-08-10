# In dataset.py, modify MapillaryDataset:
import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MapillaryDataset(Dataset):
    def __init__(self, root_dir, split='training', patch_size=512, ignore_index=255):
        images_dir = os.path.join(root_dir, split, 'images')
        masks_dir  = os.path.join(root_dir, split, 'v2.0', 'labels')
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Папка с изображениями не найдена: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"Папка с масками не найдена: {masks_dir}")
        self.img_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.mask_paths = sorted([
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.lower().endswith('.png')
        ])
        if len(self.img_paths) != len(self.mask_paths):
            raise AssertionError(f"Число изображений ({len(self.img_paths)}) и масок ({len(self.mask_paths)}) разное")
        self.patch_size = patch_size
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]) 
        img_np = np.array(img)              
        mask_np = np.array(mask, dtype=np.int64)

        h, w = mask_np.shape
        ph = min(self.patch_size, h)
        pw = min(self.patch_size, w)

        top = random.randint(0, h - ph) if h > ph else 0
        left = random.randint(0, w - pw) if w > pw else 0
        cropped_img = img_np[top:top+ph, left:left+pw]
        cropped_mask = mask_np[top:top+ph, left:left+pw]

        if ph < self.patch_size or pw < self.patch_size:
            pad_img = np.zeros((self.patch_size, self.patch_size, 3), dtype=img_np.dtype)
            pad_mask = np.full((self.patch_size, self.patch_size), fill_value=self.ignore_index, dtype=np.int64)
            pad_img[:ph, :pw, :] = cropped_img
            pad_mask[:ph, :pw] = cropped_mask
            cropped_img = pad_img
            cropped_mask = pad_mask

        img_tensor = torch.from_numpy(cropped_img.transpose(2,0,1)).float() / 255.0
        mask_tensor = torch.from_numpy(cropped_mask)
        return img_tensor, mask_tensor

class MapillaryDatasetAug(MapillaryDataset):
    def __init__(self, root_dir, split='training', patch_size=512,
                 flip_prob=0.5, rotation_deg=15, color_jitter_params=None, ignore_index=255):
        super().__init__(root_dir, split, patch_size, ignore_index=ignore_index)
        self.flip_prob = flip_prob
        self.rotation_deg = rotation_deg
        self.cj_params = color_jitter_params or {}

    def __getitem__(self, idx):
        img_tensor, mask_tensor = super().__getitem__(idx)
        img = img_tensor.numpy().transpose(1,2,0)  
        mask = mask_tensor.numpy()                

        H, W = mask.shape
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        img_pil = Image.fromarray((img*255).astype(np.uint8)).rotate(angle, resample=Image.BILINEAR)
        mask_pil = Image.fromarray(mask.astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
        img = np.array(img_pil).astype(np.float32) / 255.0
        mask = np.array(mask_pil).astype(np.int64)
        if 'brightness' in self.cj_params and random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img_pil)
            factor = 1 + random.uniform(-self.cj_params['brightness'], self.cj_params['brightness'])
            img_pil = enhancer.enhance(factor)
            img = np.array(img_pil).astype(np.float32) / 255.0
        if 'contrast' in self.cj_params and random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img_pil)
            factor = 1 + random.uniform(-self.cj_params['contrast'], self.cj_params['contrast'])
            img_pil = enhancer.enhance(factor)
            img = np.array(img_pil).astype(np.float32) / 255.0
        if 'saturation' in self.cj_params and random.random() < 0.5:
            enhancer = ImageEnhance.Color(img_pil)
            factor = 1 + random.uniform(-self.cj_params['saturation'], self.cj_params['saturation'])
            img_pil = enhancer.enhance(factor)
            img = np.array(img_pil).astype(np.float32) / 255.0

        h2, w2, _ = img.shape
        ph = min(self.patch_size, h2)
        pw = min(self.patch_size, w2)
        if h2 < self.patch_size or w2 < self.patch_size:
            pad_img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            pad_mask = np.full((self.patch_size, self.patch_size), fill_value=self.ignore_index, dtype=np.int64)
            pad_img[:h2, :w2, :] = img
            pad_mask[:h2, :w2] = mask
            img = pad_img
            mask = pad_mask

        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()
        mask_tensor = torch.from_numpy(mask)
        return img_tensor, mask_tensor
