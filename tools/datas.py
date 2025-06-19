import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalMultimodalDataset(Dataset):
    def __init__(self, image_dir, text_dir, img_size=256, augment=False):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.augment = augment
        
        # 获取所有样本ID
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 图像转换
        if augment:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])
        
        # 文本分词器
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, f"{self.ids[idx]}.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # 加载文本
        text_path = os.path.join(self.text_dir, f"{self.ids[idx]}.txt")
        with open(text_path, 'r') as f:
            text = f.read()
        
        # 图像转换
        image = self.transform(image=image)["image"]
        
        # 文本分词
        text_encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        
        # 移除batch维度
        text_input_ids = text_encoding["input_ids"].squeeze(0)
        text_attention_mask = text_encoding["attention_mask"].squeeze(0)
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text": text
        }