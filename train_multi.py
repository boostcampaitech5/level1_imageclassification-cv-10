import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import plotly.offline 
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MaskBaseDataset
from loss import create_criterion

import wandb

# -- task
task_dict = {'default': 18, 'mask': 3, 'gender': 2, 'age': 3, 'multi':0}

# -- confusion 
class_dict = {'mask': ['Wear', 'Incorrect', 'Not Wear'], 'gender': ['Male', 'Female'], 'age': ['< 30', '>= 30 and < 60', '>= 60']}

# -- evaluation
eval_dict = {'accuracy': lambda preds, labels: accuracy_score(preds, labels), 'f1': lambda preds, labels: f1_score(preds, labels, average='macro')}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join('./' + model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskSplitByProfileDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    
    num_classes = task_dict[args.task]

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: AlbumAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    # model = BaseModel(num_classes=num_classes).to(device)
    model_module = getattr(import_module("model"), args.model + "_Model")
    model = model_module(lr=args.lr).to(device)
    # model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion) 
    gender_crierion = nn.BCELoss()
    if args.criterion == 'f1' or args.criterion == 'label_smoothing':
        criterion.classes = task_dict[args.task]

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
           
    optimizer = opt_module(
        model.train_params
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    ## 수정금지 ##
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    ## 수정금지 ##

    age_best_val_acc, gender_best_val_acc, mask_best_val_acc, best_val_evaluation = [0] * 4
    age_best_val_loss, gender_best_val_loss, mask_best_val_loss, best_val_loss = [np.inf] * 4
    patience = 5
    counter = 0
    # getattr로 decode 함수 가져와서 적용 
    decode_func = getattr(import_module("dataset"), f"decode_{args.task}_class" )
    for epoch in range(args.epochs):
        # train loop
        model.train()
        mask_loss_value, gender_loss_value, age_loss_value, loss_value = [0] * 4
        mask_matches, gender_matches, age_matches, matches = [0] * 4
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            mask_labels, gender_labels, age_labels = decode_func(labels)

            optimizer.zero_grad()
            if args.log_var:
                mask_outs, gender_outs, age_outs, log_vars = model(inputs)
                precision_mask = torch.exp(-log_vars[0])
                precision_age = torch.exp(-log_vars[2])
                precision_gender = 0.1
            else:
                mask_outs, gender_outs, age_outs = model(inputs)
                precision_mask = 0.25
                precision_age = 0.5
                precision_gender = 0.25 

            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.where(gender_outs <= torch.tensor(0.5), torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
            age_preds = torch.argmax(age_outs, dim=-1)

            mask_loss = criterion(mask_outs, mask_labels)
            gender_loss = gender_crierion(gender_outs.float().squeeze(), gender_labels.float())
            age_loss = criterion(age_outs, age_labels)

            loss = precision_mask * mask_loss + precision_gender * gender_loss + precision_age * age_loss 
            preds = mask_preds * 6 + gender_preds * 3 + age_preds

            loss.backward()
            optimizer.step()

            mask_loss_value += mask_loss.item()
            gender_loss_value += gender_loss.item()
            age_loss_value += age_loss.item()
            loss_value += loss.item()

            mask_matches += eval_dict[args.evaluation](mask_preds.data.cpu(), mask_labels.data.cpu())
            gender_matches += eval_dict[args.evaluation](gender_preds.data.cpu(), gender_labels.data.cpu())
            age_matches += eval_dict[args.evaluation](age_preds.data.cpu(), age_labels.data.cpu())

            if (idx + 1) % args.log_interval == 0:
                mask_train_loss = mask_loss_value / args.log_interval
                mask_train_evaluation = mask_matches / args.log_interval
                gender_train_loss = gender_loss_value / args.log_interval
                gender_train_evaluation = gender_matches / args.log_interval
                age_train_loss = age_loss_value / args.log_interval
                age_train_evaluation = age_matches / args.log_interval

                train_loss = loss_value / args.log_interval
                train_evaluation = (mask_train_evaluation + gender_train_evaluation + age_train_evaluation) / 3

                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}] ({idx + 1}/{len(train_loader)})\t|| "
                    f"{'training':>15} {args.criterion} loss {train_loss:4.4}\t|| training {args.evaluation} {train_evaluation:4.2%}\t|| lr {current_lr}"
                )
                print(
                    f"\t\t\t\t{'gender training':>15} {args.criterion} loss {gender_train_loss:4.4}\t|| training {args.evaluation} {gender_train_evaluation:4.2%}\t|| lr {current_lr}"
                )
                print(
                    f"\t\t\t\t{'age training':>15} {args.criterion} loss {age_train_loss:4.4}\t|| training {args.evaluation} {age_train_evaluation:4.2%}\t|| lr {current_lr}"
                )
                print(
                    f"\t\t\t\t{'mask training':>15} {args.criterion} loss {mask_train_loss:4.4}\t|| training {args.evaluation} {mask_train_evaluation:4.2%}\t|| lr {current_lr}"
                )
                print()
                if args.wdb_on:
                    wandb.log({
                        f"Train/{args.criterion} loss": train_loss, f"Train/{args.evaluation}": train_evaluation,
                        f"mask Train/{args.criterion} loss": mask_train_loss, f"Train/{args.evaluation}": mask_train_evaluation,
                        f"gender Train/{args.criterion} loss": gender_train_loss, f"Train/{args.evaluation}": gender_train_evaluation,
                        f"age Train/{args.criterion} loss": age_train_loss, f"Train/{args.evaluation}": age_train_evaluation})
                
                mask_loss_value, gender_loss_value, age_loss_value, loss_value = [0] * 4
                mask_matches, gender_matches, age_matches, matches = [0] * 4

        scheduler.step()

        # --confusion matrix
        if args.confusion:
            mask_pred_conf_item = []
            gender_pred_conf_item = []
            age_pred_conf_item = []

            mask_label_conf_item = []
            gender_label_conf_item = []
            age_label_conf_item = []
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            age_val_loss_items = []
            gender_val_loss_items = []
            mask_val_loss_items = []
            val_loss_items = []

            age_val_acc_items = []
            gender_val_acc_items = []
            mask_val_acc_items = []
            val_evaluation_items = []
            figure = None

            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask_labels, gender_labels, age_labels = decode_func(labels)  

                if args.log_var:
                    mask_outs, gender_outs, age_outs, log_vars = model(inputs)
                    precision_mask = torch.exp(-log_vars[0])
                    precision_age = torch.exp(-log_vars[2])
                    precision_gender = 0.1
                else:  
                    mask_outs, gender_outs, age_outs = model(inputs) 
                    precision_mask = 0.25
                    precision_age = 0.5
                    precision_gender = 0.25 

                mask_preds = torch.argmax(mask_outs, dim=-1)
                age_preds = torch.argmax(age_outs, dim=-1)
                gender_preds = torch.where(gender_outs <= torch.tensor(0.5), torch.tensor([0.]).to(device), torch.tensor([1.]).to(device))

                mask_loss = criterion(mask_outs, mask_labels).item()
                age_loss = criterion(age_outs, age_labels).item()
                gender_loss = gender_crierion(gender_outs.float().squeeze(), gender_labels.float()).item()

                preds = mask_preds * 6 + gender_preds * 3 + age_preds
                loss = precision_mask * mask_loss + precision_gender * gender_loss + precision_age * age_loss

                mask_matches = eval_dict[args.evaluation](mask_preds.data.cpu(), mask_labels.data.cpu()).item()
                gender_matches = eval_dict[args.evaluation](gender_preds.data.cpu(), gender_labels.data.cpu()).item()
                age_matches = eval_dict[args.evaluation](age_preds.data.cpu(), age_labels.data.cpu()).item()
                evaluation_item = (mask_matches + gender_matches + age_matches) / 3

                if args.confusion:
                    mask_pred_conf_item.extend(mask_preds.data.cpu().numpy())
                    gender_pred_conf_item.extend(gender_preds.data.cpu().numpy())
                    age_pred_conf_item.extend(age_preds.data.cpu().numpy())

                    mask_label_conf_item.extend(mask_labels.data.cpu().numpy())
                    gender_label_conf_item.extend(gender_labels.data.cpu().numpy())
                    age_label_conf_item.extend(age_labels.data.cpu().numpy())

                mask_val_loss_items.append(mask_loss)
                mask_val_acc_items.append(mask_matches)
                gender_val_loss_items.append(gender_loss)
                gender_val_acc_items.append(gender_matches)
                age_val_loss_items.append(age_loss)
                age_val_acc_items.append(age_matches)
                val_loss_items.append(loss)
                val_evaluation_items.append(evaluation_item)

            if args.confusion:
                
                mask_conf_matrix = confusion_matrix(mask_label_conf_item, mask_pred_conf_item)
                gender_conf_matrix = confusion_matrix(gender_label_conf_item, gender_pred_conf_item)
                age_conf_matrix = confusion_matrix(age_label_conf_item, age_pred_conf_item)

                mask_df_cm = pd.DataFrame(mask_conf_matrix / np.sum(mask_conf_matrix, axis=1)[:, None], 
                                     index = [i for i in class_dict['mask']], columns = [i for i in class_dict['mask']])
                gender_df_cm = pd.DataFrame(gender_conf_matrix / np.sum(gender_conf_matrix, axis=1)[:, None], 
                                     index = [i for i in class_dict['gender']], columns = [i for i in class_dict['gender']])
                age_df_cm = pd.DataFrame(age_conf_matrix / np.sum(age_conf_matrix, axis=1)[:, None], 
                                     index = [i for i in class_dict['age']], columns = [i for i in class_dict['age']])
                
                fig, ax = plt.subplots(figsize = (12,7), ncols=3)
                mask_heatmap = sn.heatmap(data=mask_df_cm, annot=True, cmap='Reds', ax=ax[0], xticklabels=mask_df_cm.columns, yticklabels=mask_df_cm.columns)
                gender_heatmap = sn.heatmap(data=gender_df_cm, annot=True, cmap='Blues', ax=ax[1], xticklabels=gender_df_cm.columns, yticklabels=gender_df_cm.columns)
                age_heatmap = sn.heatmap(data=age_df_cm, annot=True, cmap='Greens', ax=ax[2], xticklabels=age_df_cm.columns, yticklabels=age_df_cm.columns)
                
                if args.wdb_on:
                    wandb.log({
                        "Confusion Matrix": wandb.Image(plt)})
                plt.close('all')
                plt.clf()
                print(mask_df_cm)  
                print(gender_df_cm)
                print(age_df_cm)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_evaluation = np.sum(val_evaluation_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)

            if val_evaluation > best_val_evaluation:
                print(f"New best model for val {args.evaluation} : {val_evaluation:4.2%}! saving the best model dict..")
                best_model = {
                        "model": copy.deepcopy(model.state_dict()),
                        "path": f"./{save_dir}/{epoch+1:03}_{args.evaluation}_{val_evaluation:4.2%}.pth"
                    }
                best_val_evaluation = val_evaluation
                counter = 0
            else:
                counter += 1
            
            if counter > patience:
                print("Early Stopping...")
                break
            
            print(
                f"[Val] {args.evaluation} : {val_evaluation:4.2%}, {args.criterion} loss: {val_loss:4.2}\t|| "
                f"best {args.evaluation} : {best_val_evaluation:4.2%}, best {args.criterion} loss: {best_val_loss:4.2}"
            )
            if args.wdb_on:
                wandb.log({
                        f"Val/ {args.criterion} loss": val_loss, f"Val/{args.evaluation}": val_evaluation
                        })
                
            
            print()

    torch.save(best_model['model'], best_model['path'])        


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='AlbumAugmentation', help='data augmentation type (default: AlbumAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=30, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--wdb_on', type=bool, default=False, help='turn on wandb(if you set True)')
    parser.add_argument('--task', type=str, default='multi', help='select task you want(default: multi) ex: [mask, gender, age, multi]')
    parser.add_argument('--confusion', type=bool, default=False, help='make confusion matrix about each task, logging on wandb')
    # loss 말고 acc, f1 선택할 수 있는 기능
    parser.add_argument('--evaluation', type=str, default='accuracy', help='set evaluation function (accuracy, f1)')

    # 임시 
    parser.add_argument('--logvar', type=bool, default=False, help='set logvar')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)
    assert not ((args.confusion == True) and (args.task == 'default')), 'confusion and default can\'t exist at the same project'
    if args.wdb_on:
        wandb.init(
            project="Mask image Classification Competition(Multi)",
            notes="Gender에 BCELoss 적용, 코드 수정",
            config={
                "img_size": args.resize,
                "loss": args.criterion,
                "eval": args.evaluation,
                "learning_rate": args.lr,
                "architecture": args.model,
                "epochs": args.epochs
            }
        )
        wandb.run.name = args.name
        wandb.run.save()
        

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
