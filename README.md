# level1_imageclassification-cv-10
level1_imageclassification-cv-10 created by GitHub Classroom

Readme author: Kyungbong Lee 

based on baseline_v2

## 1. Train 
---
```
python train.py --augmentation AlbumAugmentation --model efficientnetv2_rw_m --criterion focal --name {name} --log_interval 30 --optimizer Adam --epochs 10
```
### 1-1. To use `wandb`
if you turn on your wandb, set parameter 
```
--wdb_on True
```
you can set wandb name

`--model {tag}`  
Example case, tag='efficientnetv2_rw_m'

`wandb.run.name = {tag}` 



## 2. Inference (make output.csv) 
---
```
python inference.py --model efficientnetv2_rw_m --model_dir ./model/{name}
```

you can see terminal message
```
Inference Done! Inference result saved at ./output/output.csv
```

## 3. dir structure
---
```
main
├───train.py
├───loss.py
├───model.py
├───dataset.py
├───inference.py    
├───output
│     └───output.csv
└───model
     └───{name}
          ├───{fold}_{epoch}_accuracy_{accuracy}.pth
          ├───{fold}_{epoch}_accuracy_{accuracy}.pth
          ├───{fold}_{epoch}_accuracy_{accuracy}.pth
          ├───{fold}_{epoch}_accuracy_{accuracy}.pth
          └───{fold}_{epoch}_accuracy_{accuracy}.pth
```
