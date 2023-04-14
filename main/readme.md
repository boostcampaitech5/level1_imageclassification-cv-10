# level1_imageclassification-cv-10
level1_imageclassification-cv-10 created by GitHub Classroom

author: Kyungbong Lee 

based on baseline_v2

## Train (Stratified K-Fold Train)
---
```
python train.py --augmentation AlbumAugmentation --model efficientnetv2_rw_m --criterion focal --name {name} --log_interval 30 --optimizer Adam --epochs 10
```
### To use `wandb`
if you turn on your wandb, set parameter 
```
--wdb_on True
```
you can set wandb name

`--model {tag}`  
Example case, tag='efficientnetv2_rw_m'

`wandb.run.name = {tag}` 


```
baseline_v2
└───{name}
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    ├───{fold}_{epoch}_accuracy_{accuracy}.pth
    └───{fold}_{epoch}_accuracy_{accuracy}.pth
```

## Inference (make output.csv) 
---
```
python inference.py --model efficientnetv2_rw_m --model_dir model/{name}
```

you can see terminal message
```
Inference Done! Inference result saved at ./output/output.csv
```