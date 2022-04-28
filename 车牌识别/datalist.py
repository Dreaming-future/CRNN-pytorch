# -*- coding: utf-8 -*-
"""
author:LTH
data:
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import GetConfig

data_transform = transforms.Compose([
    transforms.Resize([24, 94]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RecTextLineDataset(Dataset):
    def __init__(self, data, type="train"):
        self.args = GetConfig()
        self.alphabets = self.get_alphabets(self.args.alphabet)
        self.str2idx = {c: i for i, c in enumerate(self.alphabets)}
        self.labels = []

        for line in data:
            length = len(line.split('\\')[-1].split('.')[0])
            # if length == 8 or length == 9 or length==7 or length==6:
            if length == 8 or length == 9 or length == 7:
                self.labels.append((line, line.split('\\')[-1].split('.')[0]))

        self.type = type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path, target = self.labels[index]

        img = Image.open(img_path).convert("RGB")

        label_encode = []

        for c in target:
            label_encode.append(self.str2idx[c])
        return img, label_encode, len(target), target

    @staticmethod
    def get_alphabets(alphabet):
        f = open(alphabet, 'r', encoding='utf-8')
        alphabets = ''.join([s.strip('\n') for s in f.readlines()])
        alphabets = ' ' + alphabets
        return alphabets


def rec_collate(batch):
    imgs = []
    labels = []
    lengths = []
    labels_word = []

    for _, sample in enumerate(batch):
        img, label, length, words = sample
        img = data_transform(img)
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        labels_word.append(words)

    labels = np.array(labels).flatten().astype(np.int)

    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths, labels_word
