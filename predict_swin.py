"""
Author: HanChen
Date: 21.06.2022
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import argparse
import json

import numpy as np
import pandas as pd
import torchvision.models as models

from transforms import build_transforms
from metrics import get_metrics
from data_utils import images_Dataloader
from network.models import get_swin_transformers

import os

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating network')
           
    # parser.add_argument('--root_path', default='/data/linyz/DFGC2022/DFGC2022_Test/images',
                        # type=str, help='path to Evaluating dataset')              

    parser.add_argument('--root_path', default='/data/linyz/DFGC2022/DFGC2022_Test/images_mtcnn',
                        type=str, help='path to Evaluating dataset')       
    parser.add_argument('--save_path', type=str, default='./save_result')           
    parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384_in22k')          
    parser.add_argument('--pre_trained', type=str, default='./save_result/models/efficientnet-b4.pth')           
    parser.add_argument('--output_txt', type=str, default='./save_result/pred_efn.txt')    
    parser.add_argument('--gpu_id', type=int, default=6)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--adjust_lr_iteration', type=int, default=500)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--val_batch_size', type=int, default=16)
    args = parser.parse_args()
    return args



def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


def test_model(model, dataloaders):
    prediction = np.array([])
    model.train(False)
    for images in dataloaders:
        input_images = Variable(images.cuda())
        outputs = model(input_images)
        pred_ = torch.nn.functional.softmax(outputs, dim=-1)
        pred_ = pred_.cpu().detach().numpy()[:, 1]

        prediction = np.insert(prediction, 0, pred_)
    return prediction

def main():
    args = parse_args()
    test_videos = os.listdir(args.root_path)

    transform_train, transform_test = build_transforms(args.resolution, args.resolution, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    # create model
    model = get_swin_transformers(model_name=args.model_name, num_classes=args.num_class)
    # load saved model
    model = load_network(model, args.pre_trained).cuda()
    model.train(False)
    model.eval()

    result_txt = open(args.output_txt, 'w', encoding='utf-8')

    for idx, video_name in enumerate(test_videos):
        video_path = os.path.join(args.root_path, video_name)
        test_dataset = images_Dataloader(video_path, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size,
            drop_last=False, shuffle=False, num_workers=0, pin_memory=False)

        with torch.no_grad():
            prediction = test_model(model, test_loader)

        if len(prediction) != 0:
            video_prediction = np.mean(prediction, axis=0)
        else:
            video_prediction = 0.5  # default is 0.5
        print(video_name + '  is  fake' if np.round(video_prediction) == 1
              else video_name + '  is  real')
        print('Probs %f' % video_prediction)


        result_txt.write(video_name + ', %f' % video_prediction + '\n')



if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    main()

