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
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset

import os


def extract_frames(videos_path, detector=None, frame_subsample_count=30, scale=1.3):
    assert detector is not None, 'model is None'

    reader = cv2.VideoCapture(videos_path)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 32
    rgb_frames = OrderedDict()
    pil_frames = OrderedDict()
    for i in range(frames_num):
        for _ in range(frame_subsample_count):
            reader.grab()

        success, frame = reader.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BRG2RGB
        rgb_frames[i] = frame
        frame = Image.fromarray(frame)  # To numpy array
        frame = frame.resize(size=[s // 2 for s in frame.size])
        pil_frames[i] = frame

    rgb_frames = list(rgb_frames.values())
    pil_frames = list(pil_frames.values())
    reader.release()
    crops = []
    for i in range(0, len(pil_frames), batch_size):
        batch_boxes, batch_probs, batch_points = detector.detect(pil_frames[i:i + batch_size], landmarks=True)
        None_array = np.array([], dtype=np.int16)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                pass
            else:
                batch_boxes[index] = None_array
        batch_boxes, batch_probs, batch_points = detector.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                       pil_frames[i:i + batch_size],
                                                                       method="probability")
        # print(batch_probs.shape)
        # print(batch_points.shape)
        # batch_boxes = np.squeeze(batch_boxes, 1)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]  # resize the box
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(rgb_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(rgb_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                crop = rgb_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]
                crops.append(crop)
            else:
                pass
    return crops


class video_Dataloader(Dataset):
    def __init__(self, videos_path, batch_size=32, transform=None, num_class=2, scale=1.3, 
                 frame_subsample_count=30, detector=None):
        assert detector is not None, 'model is None'
        self.videos_path = videos_path

        # extract face images, all face images are rgb images
        self.face_images = extract_frames(self.videos_path, detector=detector, scale=scale,
                                          frame_subsample_count=frame_subsample_count)

        self.batch_size = batch_size
        self.num_class = num_class
        self.transform = transform

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)


class images_Dataloader(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.face_images = self.load_image()
        self.transform = transform

    def load_image(self):
        face_images = []
        for i, filename in enumerate(glob.glob(self.video_path + '/*')):
            if filename.find('json') == -1: 
                if int(filename.split('/')[-1].replace('.png', '')) % 10 == 0:
                    try:
                        frame = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                        face_images.append(frame)
                    except:
                        print('something wrong!!!')

        return face_images

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)


class images_Dataloader_all(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.face_images = self.load_image()
        self.transform = transform

    def load_image(self):
        face_images = []
        for i, filename in enumerate(glob.glob(self.video_path + '/*')):
            if filename.find('json') == -1: 
                try:
                    frame = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    face_images.append(frame)
                except:
                    print('something wrong!!!')

        return face_images

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)


class faces_Dataloader(Dataset):
    def __init__(self, face_images, transform=None):
        self.face_images = face_images
        self.transform = transform

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)

