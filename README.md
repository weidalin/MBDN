## Multi-branch Detection Network based on Trigger Attention for Pedestrian Detection under Occlusion
Center and Scale Prediction (CSP) first introduced the Anchor-free method to the field of pedestrian detection. Pedestrian detection often occurs in complex scenes subject to occlusion, and it is difficult to extract pedestrian features in a single centre point prediction in CSP. To solve this problem, this paper presents a multi-branch detection network (MBDN) based on trigger attention. Firstly, a multi-centre point prediction branch feature extraction model (multi-centre) is proposed to solve the problem of CSP missed detections in occlusion scenarios. Secondly, a novel trigger attention module is designed. The module uses visible parts as triggers to automatically learn the weight relationships of multiple branches, let the network automatically learn the confidence of the centre points of different branches, and automatically strengthen the branch where the visible area on the feature map is located. Finally, a channel non-maximum suppression (NMS) module is used in the MBDN network to reduce the redundant bounding boxes. Then experiments results show that the log-average missing rate (MR^(-2)) of the heavy subset is reduced from 49.63% to 45.51% while maintaining the performance on a reasonable subset.
![avatar](./fig1.bmp)

## This code is forked from [ACSP](https://github.com/WangWenhao0716/Adapted-Center-and-Scale-Prediction.git) ，and modified on the basis of ACSP



## train
- config setting 
train.py 
```
python train.py
```

## test
- config setting 
test.py 
```
python test.py
```
## model 
Put `ACSP_150.pth.tea` into `models/V42_resnetv2sn101_headandfullvisible3center3gaussmap_triggerat_originalgausspointmutiyy1103add08_640_1280_2gpuper1img_lr0.0001/ckpt
` folder

Put `pretrained` into `data` folder

[model link](https://pan.baidu.com/s/1wFppepz48cK7OxM9rcKEzg?pwd=6k9r)

