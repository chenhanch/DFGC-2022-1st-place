"""
Some codes borrowed from https://github.com/jphdotam/DFDC/blob/master/cnn3d/training/datasets_video.py
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 13.10.2020
"""

import cv2
import math
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import os


class binary_Rebalanced_Dataloader(object):
    def __init__(self, video_names=[], video_labels=[], phase='train', num_class=2, transform=None):
        assert phase in ['train', 'valid', 'test']
        self.video_names = video_names
        self.video_labels = video_labels
        self.phase = phase
        self.num_classes = num_class
        self.transform = transform
        self.default_video_name = '/data/linyz/Celeb-DF-v2/face_crop_png/Celeb-real/id53_0008.mp4'
        self.default_label = 0

    def __getitem__(self, index):
        try:
            video_name = self.video_names[index]
            label = self.video_labels[index]
            image_name = random.sample(os.listdir(video_name), 1)[0]
            image_path = os.path.join(video_name, image_name)
        except:
            print(video_name)
            label = self.default_label
            image_name = random.sample(os.listdir(self.default_video_name), 1)[0]
            image_path = os.path.join(self.default_video_name, image_name)        
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image, label

    def __len__(self):
        return len(self.video_names)


