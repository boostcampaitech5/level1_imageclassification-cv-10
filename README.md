# level1_imageclassification-cv-10
level1_imageclassification-cv-10 created by GitHub Classroom

Readme author: Kyungbong Lee 

based on baseline_v2

## 1. Train 
---
```
python train.py --augmentation AlbumAugmentation --model efficientnetv2_rw_m --criterion focal --name {name} --log_interval 30 --optimizer Adam --epochs 10
```
after Merge pull request #16
```
python train.py --augmentation AlbumAugmentation \
--model efficientnetv2_rw_m --criterion f1 --name {name} \
--log_interval 30 --optimizer Adam --epochs 20 \
--task {default or single task name} \
--confusion True --wdb_on True --evaluation f1
```
if you want Multitask learning:
```
python train_multi.py --model multi_efficientnetv2_rw_m \
--name {name} --task multi --criterion focal \
--epochs 20
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

### 1-2. To Train each single task 
Set parameter
```
--task {task_name}
```
task_name:

> 'default': default setting (18 classes)

> 'age' or 'gender' or 'mask': each single task train (each class num)

### 1-3. To logging Confusion Matrix 
Set parameter 
```
--confusion True
```
But, when you set parameter `--confusion True` and `--task default` same time,
```
(args.confusion == True) and (args.task == 'default')
```
by this code raise error 

### 1-4. Setting Callback condition 
Set parameter 
```
--evaluation {accuracy or f1}
```
In previous update(before merge pull request #16), callback option is fixed "accuracy"

## 2. Inference (make output.csv) 
---
you must set train prameter `--task default` 
 
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
          └───{epoch}_evaluation_{best_val_evaluation}.pth
          
```
