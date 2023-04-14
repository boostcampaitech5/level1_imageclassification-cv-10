# level1_imageclassification-cv-10
level1_imageclassification-cv-10 created by GitHub Classroom

author: Kyungbong Lee 

based on baseline_v2

## Train (Stratified K-Fold Train)
---
```
python train.py --augmentation AlbumAugmentation --model tf_efficientnet_b7 --criterion focal --name {name} --log_interval 30 --optimizer Adam --epochs 10
```
if you turn on your wandb, set parameter 
```
--wdb_on 
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
python inference.py --model tf_efficientnet_b7 --model_dir model/{name}
```

you can see terminal message
```
Inference Done! Inference result saved at ./output/output.csv
```