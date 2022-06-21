# [DFGC2022](https://codalab.lisn.upsaclay.fr/competitions/3523#learn_the_details-overview) Detection Solution

This repo provides an solution for the DeepFake Game Competition (DFGC) @ IJCB 2022 Detection track. Our solution achieve the 1st in the final phase of the DFGC Detection track. The ranking can be seen [here](https://codalab.lisn.upsaclay.fr/competitions/3523#learn_the_details-evaluation)

## 1. Authors

Institution: Shenzhen Key Laboratory of Media Information Content Security([MICS](http://media-sec.szu.edu.cn/))

Adviser: [Professor Bin Li](http://media-sec.szu.edu.cn/view/libin105.html) 

Username: HanChen

Team members:
- [Han Chen](https://github.com/chenhanch)
- [Baoying Chen](https://github.com/beibuwandeluori) 
- [Linhui Hu](https://github.com/LinhuiHu)
- [Qiushi Li](https://github.com/Harvest-Li)
- [Yuzhen Lin](https://github.com/Linyuzhen)

## 2. A brief report
- **Model structure**：ConvNext（convnext_xlarge_384_in22ft1k）and SwinTransformer(swin_large_patch4_window12_384_in22k), with weights pretrained on ImageNet dataset。

- **Ensemble methods**：Two ConvNext at different epochs and one Swin-Transformer.

- **Augmentation methods**：HorizontalFlip、GaussNoise、GaussianBlur

- **Data processing**：A face detector MTCNN is used to crop the face images from video frame (enlarged the face region by a factor of 1.3). Resize the input shape to (3,384,384).

- **Training losses**：BCELoss

- **The dataset used for training**：FF++(c23 and c40)[1]/(DeepFake、Face2Face、FaceSwap、FaceShifter、NeuralTextures)、FF++(c23)/HifiFace[2]、UADFV[3]、DF-TIMIT[4]、DeeperForensics-1.0[5]、DeepFakeDection(c23 and c40)[6]、Celeb-DF[7]、WildDeepfake[8]、DFDC[9]

- **Tricks **：Don't use smaller size input、Early stop training、Larger model

- References and open-source resources.

  [1] FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV 2019

  [2] https://johann.wang/HifiFace/

  [3] In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking. WIFS 2018.

  [4] Vulnerability Assessment and Detection of Deepfake Videos. ICB 2019

  [5] DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection

  [6] https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html

  [7] Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics. CVPR2020.

  [8] WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection. 2020 ACM MM.

  [9] The DeepFake Detection Challenge (DFDC) Dataset. Arxiv 2020.

## 3. Training Code

### 3.1 Extract faces from video and save as png (Only used to extract faces from the test set video of the competition)

```sh
python extract_video_mtcnn.py --input_root_path <video_path> --output_root_path <saved_image_path> --gpu_id <GPU_ID>
```

### 3.2 Training

```sh
## train swin_large_patch4_window12_384_in22k
python train_swin.py --model_name swin_large_patch4_window12_384_in22k --resolution 384 --gpu_id <GPU_ID>

## train convnext_xlarge_384_in22ft1k
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_convnext.py --model_name convnext_xlarge_384_in22ft1k --resolution 384
```

### 3.3 Testing

```sh
## test the model weight of the 40th epoch of swin_large_patch4_window12_384_in22k
python predict_swin.py --model_name swin_large_patch4_window12_384_in22k --resolution 384 --pre_trained ./save_result/models/swin_large_patch4_window12_384_in22k_40.pth --root_path <saved_image_path> --output_txt ./save_result/pred_swin_large_patch4_window12_384_in22k_40e.txt --gpu_id <GPU_ID>

## test the model weight of the 10th epoch of convnext_xlarge_384_in22ft1k
python predict_convnext.py --model_name convnext_xlarge_384_in22ft1k --resolution 384 --pre_trained ./save_result/models/convnext_xlarge_384_in22ft1k_10.pth --root_path <saved_image_path> --output_txt ./save_result/pred_convnext_xlarge_384_in22ft1k_10e.txt --gpu_id <GPU_ID>

## test the model weight of the 30th epoch of convnext_xlarge_384_in22ft1k
python predict_convnext.py --model_name convnext_xlarge_384_in22ft1k --resolution 384 --pre_trained ./save_result/models/convnext_xlarge_384_in22ft1k_30.pth --root_path <saved_image_path> --output_txt ./save_result/pred_convnext_xlarge_384_in22ft1k_30e.txt --gpu_id <GPU_ID>
```

### 3.4 ensemble

```sh
## ensemble
python merge_csv.py --output_txt preds.txt
```

## 4. Environment

```
facenet-pytorch==2.5.0
torch==1.9.0
dlib==19.21.1
```
