# level1_imageclassification-cv-10
level1_imageclassification-cv-10 created by GitHub Classroom

author: Kyungbong Lee 

based on baseline_v2

## Train (Stratified K-Fold Train)
---
```
python train.py --augmentation AlbumAugmentation --model build_model --criterion focal --name m2 --log_interval 30 --optimizer Adam --epochs 10
```

```
baseline_v2
└───m2
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    └───{fold}_{epoch}_accuracy_{accuracy}.pth
```

## Inference (make output.csv) 
---
```
python inference.py --model build_model --model_dir model/m2
```

you can see terminal message
```
Inference Done! Inference result saved at ./output/output.csv
```