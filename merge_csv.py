import warnings
warnings.filterwarnings("ignore")

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ff_df_dataset')                 
    parser.add_argument('--output_txt', type=str, default='preds.txt')        
    args = parser.parse_args()
    return args


args = parse_args()

from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict


if __name__ == '__main__':
    txt_files = ['./save_result/txt/pred_swin_large_patch4_window12_384_in22k_40e.txt',
                './save_result/txt/pred_convnext_xlarge_384_in22ft1k_10e.txt',
                './save_result/txt/pred_convnext_xlarge_384_in22ft1k_30e.txt',]
    weights = [0.2, 0.3, 0.5]
    dicts = []
    for txt_file in txt_files:
        dic = {}
        with open(txt_file, 'r') as f:
            videos_names = f.readlines()
            for i in videos_names:
                name = i.strip().split(', ')[0]
                probs = float(i.strip().split(', ')[1])
                dic[name] = probs
        dicts.append(dic)

    results = OrderedDict()
    for key, value in dicts[0].items():
        results[key] = value * weights[0]
        results[key] += dicts[1][key] * weights[1]
        results[key] += dicts[2][key] * weights[2]

    result_txt = open(args.output_txt, 'w', encoding='utf-8')
    for key, value in results.items():
        result_txt.write(key + ', %f' % value + '\n')

