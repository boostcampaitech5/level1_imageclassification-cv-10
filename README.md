# level1_imageclassification-cv-10
level1_imageclassification-cv-10 created by GitHub Classroom

Readme author: Kyungbong Lee 

based on baseline_v2

## 1. Train 
---
### 1-1. default train (num_classes=18)
```
python train.py --model {model_name} --name {name}
```
### 1-2. Single task train (each task:['mask', 'gender', 'age'])
```
python train.py --model {model_name} --name {name} --task {task_name}
```
### 1-3. Multi task train 
```
python train_multi.py --model multi_{model_name} --name {name}
```
## 2. Inference (make output.csv) 
---
Single task inference Not supported. 

### 2-1. default inference 
```
python inference.py --model {model_name} --model_dir ./model/{name}
```
### 2-2 multi task inference 
```
python inference_multi.py --model multi_{model_name} --model_dir ./model/{name}
```
multi task inference case: 
Can get `output_default.csv(18 classes)`, `output_multi.csv(each task)` 

## 3. Additional 
### 3-1. To use `wandb`
if you turn on your wandb, set parameter 
```
--wdb_on True
```
you can set wandb name

`--model {model_name}`  

### 3-2. To logging Confusion Matrix 
Set parameter 
```
--confusion True
```
you can see in terminal 
```
                    < 30  >= 30 and < 60     >= 60
< 30            0.904342        0.093623  0.002035
>= 30 and < 60  0.052554        0.900814  0.046632
>= 60           0.005714        0.400000  0.594286
```
But, when you set parameter `--confusion True` and `--task default` same time,
```
(args.confusion == True) and (args.task == 'default')
```
by this code raise error 

### 3-3. Setting parameters
You can set the various parameters you want.

`--augmentation`: set_transform에 의해 정해지는 augmentation 기법 (default: AlbumAugmentation)

`--optimizer`: torch.optim에 존재하는 optimizer(default: Adam)

`--criterion`: loss함수 / loss.py에 존재하는 loss 함수 사용 가능 (default: focal Loss)

`--evaluation`: best model 선정 기준 / accuracy, f1 score (default: accuracy)


## 4. dir structure
---
```
main
├───train.py
├───loss.py
├───model.py
├───dataset.py
├───inference.py    
├───output
│     ├───output_default.csv
│     └───output_multi.csv
└───model
     └───{name}
          └───{epoch}_{evaluation}_{best_val_evaluation}.pth
          
```
