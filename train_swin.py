"""
Author: HanChen
Date: 21.06.2022
"""
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from logger import create_logger

from network.models import get_swin_transformers
from transforms import build_transforms
from metrics import get_metrics
from dataset import binary_Rebalanced_Dataloader

import os



######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network

def parse_args():
    parser = argparse.ArgumentParser(description='Training network')
           
    parser.add_argument('--root_path_dfdc', default='/data/linyz/DFDC/face_crop_png',
                        type=str, help='path to DFDC dataset')         
    parser.add_argument('--save_path', type=str, default='./save_result')           
    parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384_in22k')           
    parser.add_argument('--gpu_id', type=int, default=6)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--adjust_lr_iteration', type=int, default=30000)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--base_lr', type=float, default=0.00005)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--val_batch_size', type=int, default=128)
    args = parser.parse_args()
    return args

def load_txt(txt_path='./txt', logger=None):
    txt_names = os.listdir(txt_path)
    tmp_videos, tmp_labels = [], []
    for txt_name in txt_names:
        with open(os.path.join(txt_path, txt_name), 'r') as f:
            videos_names = f.readlines()
            for i in videos_names:
                if i.find('landmarks') != -1:
                    continue
                if len(os.listdir(i.strip().split()[0])) == 0:
                    continue
                tmp_videos.append(i.strip().split()[0])
                tmp_labels.append(int(i.strip().split()[1]))
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        print(timeStr, len(tmp_labels), sum(tmp_labels), sum(tmp_labels)/len(tmp_labels))    
    return tmp_videos, tmp_labels


def main():
    args = parse_args()

    logger = create_logger(output_dir='%s/report' % args.save_path, name=f"{args.model_name}")
    logger.info('Start Training %s' % args.model_name)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    logger.info(timeStr)  

    transform_train, transform_test = build_transforms(args.resolution, args.resolution, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    train_videos, train_labels = [], []
    for idx in tqdm(range(0, 50)):
        sub_name = 'dfdc_train_part_%d' % idx
        video_sub_path = os.path.join(args.root_path_dfdc, sub_name)
        with open(os.path.join(video_sub_path, 'metadata.json')) as metadata_json:
            metadata = json.load(metadata_json)
        for key, value in metadata.items(): 
            if value['label'] == 'FAKE': # FAKE or REAL
                label = 1
            else:
                label = 0
            inputPath = os.path.join(args.root_path_dfdc, sub_name, key)
            if len(os.listdir(inputPath)) == 0:
                continue
            train_videos.append(inputPath)
            train_labels.append(label)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    print(timeStr, len(train_labels), sum(train_labels), sum(train_labels)/len(train_labels))
    
    tmp_videos, tmp_labels = load_txt(txt_path='./txt')
    train_videos += tmp_videos
    train_labels += tmp_labels
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    print(timeStr, len(train_labels), sum(train_labels), sum(train_labels)/len(train_labels))

    train_dataset = binary_Rebalanced_Dataloader(video_names=train_videos, video_labels=train_labels, phase='train', 
                                                num_class=args.num_class, transform=transform_train)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    print(timeStr, 'All Train videos Number: %d' % (len(train_dataset)))

    model = get_swin_transformers(model_name=args.model_name, num_classes=args.num_class).cuda()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, num_workers=6, pin_memory=True)


    loss_name = ['BCE']
    iteration = 0
    running_loss = {loss: 0 for loss in loss_name}
    for epoch in range(args.num_epochs):
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        logger.info(timeStr+'Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        logger.info(timeStr+'-' * 10)

        model.train(True)  # Set model to training mode
        # Iterate over data (including images and labels).
        for index, (images, labels) in enumerate(train_loader):
            iteration += 1
            # wrap them in Variable
            images = Variable(images.cuda().detach())
            labels = Variable(labels.cuda().detach())

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)

            # update the parameters
            loss.backward()
            optimizer.step()

            running_loss['BCE'] += loss.item()
            # break
            if iteration % 100 == 0:
                timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
                logger.info(timeStr+'Epoch: {:g}, Itera: {:g}, Step: {:g}, BCE: {:g} '.
                      format(epoch, index, len(train_loader), *[running_loss[name] / 100 for name in loss_name]))
                running_loss = {loss: 0 for loss in loss_name}
            
            if iteration % args.adjust_lr_iteration == 0:
                scheduler.step()
                
        if epoch % 20 == 0:
            timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
            logger.info(timeStr + '  Save  Model  ')
            save_network(model, '%s/models/%s_%d.pth' % (args.save_path, args.model_name, epoch))
            
    save_network(model, '%s/models/%s.pth' % (args.save_path, args.model_name))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path)

    main()
