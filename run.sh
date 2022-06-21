#!/bin/bash


## train swin_large_patch4_window12_384_in22k
python train_swin.py --model_name swin_large_patch4_window12_384_in22k --resolution 384 --gpu_id 0
## test the model weight of the 40th epoch of swin_large_patch4_window12_384_in22k
python predict_swin.py --model_name swin_large_patch4_window12_384_in22k --resolution 384 --pre_trained ./save_result/models/swin_large_patch4_window12_384_in22k_40.pth --root_path /data/linyz/DFGC2022/DFGC2022_Test/images_mtcnn --output_txt ./save_result/pred_swin_large_patch4_window12_384_in22k_40e.txt --gpu_id 0

## train convnext_xlarge_384_in22ft1k
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_convnext.py --model_name convnext_xlarge_384_in22ft1k --resolution 384
## test the model weight of the 10th epoch of convnext_xlarge_384_in22ft1k
python predict_convnext.py --model_name convnext_xlarge_384_in22ft1k --resolution 384 --pre_trained ./save_result/models/convnext_xlarge_384_in22ft1k_10.pth --root_path /data/linyz/DFGC2022/DFGC2022_Test/images_mtcnn --output_txt ./save_result/pred_convnext_xlarge_384_in22ft1k_10e.txt --gpu_id 0
## test the model weight of the 30th epoch of convnext_xlarge_384_in22ft1k
python predict_convnext.py --model_name convnext_xlarge_384_in22ft1k --resolution 384 --pre_trained ./save_result/models/convnext_xlarge_384_in22ft1k_30.pth --root_path /data/linyz/DFGC2022/DFGC2022_Test/images_mtcnn --output_txt ./save_result/pred_convnext_xlarge_384_in22ft1k_30e.txt --gpu_id 0

## ensemble
python merge_csv.py --output_txt preds.txt