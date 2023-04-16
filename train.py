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
task_dict = {'default': 18, 'mask': 3, 'gender': 2, 'age': 3}

# -- confusion 
class_dict = {'mask': ['Wear', 'Incorrect', 'Not Wear'], 'gender': ['Male', 'Female'], 'age': ['< 30', '>= 30 and < 60', '>= 60']}

# -- evaluation
eval_dict = {'accuracy': lambda preds, labels: accuracy_score(preds, labels), 'f1': lambda preds, labels: f1_score(preds, labels, average='weighted')}

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
    model = model_module(num_classes=num_classes, lr=args.lr).to(device)
    # model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion) 
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

    best_val_evaluation = 0
    best_val_loss = np.inf
    patience = 5
    counter = 0
    # getattr로 decode 함수 가져와서 적용 
    decode_func = getattr(import_module("dataset"), f"decode_{args.task}_class" )
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch

            labels = decode_func(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += eval_dict[args.evaluation](preds.data.cpu(), labels.data.cpu())
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_evaluation = matches / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}] ({idx + 1}/{len(train_loader)})\t|| "
                    f"training {args.criterion} loss {train_loss:4.4}\t|| training {args.evaluation} {train_evaluation:4.2%}\t|| lr {current_lr}"
                )
                
                if args.wdb_on:
                    wandb.log({
                        f"Train/{args.criterion} loss": train_loss, f"Train/{args.evaluation}": train_evaluation})
                
                loss_value = 0
                matches = 0

        scheduler.step()

        # --confusion matrix
        if args.confusion:
            pred_conf_item = []
            label_conf_item = []
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_evaluation_items = []
            figure = None

            for val_batch in val_loader:
                inputs, labels = val_batch
                labels = decode_func(labels)        # torch.Tensor([0, 1, 1, 1, 0, 0, 2, 0... ])
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)                
                preds = torch.argmax(outs, dim=-1)  # torch.Tensor([0, 1, 2, 0, 0, 0, 2, 0... ])

                loss_item = criterion(outs, labels).item()
                evaluation_item = eval_dict[args.evaluation](preds.data.cpu(), labels.data.cpu())
                print(evaluation_item)
                if args.confusion:
                    pred_conf_item.extend(preds.data.cpu().numpy())
                    label_conf_item.extend(labels.data.cpu().numpy())

                val_loss_items.append(loss_item)
                val_evaluation_items.append(evaluation_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
            if args.confusion:
                
                conf_matrix = confusion_matrix(label_conf_item, pred_conf_item)
                df_cm = pd.DataFrame(conf_matrix / np.sum(conf_matrix, axis=1)[:, None], 
                                     index = [i for i in class_dict[args.task]], columns = [i for i in class_dict[args.task]])
                plt.subplots(figsize = (12,7))
                s = sn.heatmap(data=df_cm, annot=True, cmap='Reds', xticklabels=df_cm.columns, yticklabels=df_cm.columns)
                if args.wdb_on:
                    wandb.log({
                        "Confusion Matrix": wandb.Image(plt)})
                plt.close('all')
                plt.clf()
                print(df_cm)
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_evaluation = np.sum(val_evaluation_items) / len(val_evaluation_items)
            best_val_loss = min(best_val_loss, val_loss)

            if val_evaluation > best_val_evaluation:
                print(f"New best model for val {args.evaluation} : {val_evaluation:4.2%}! saving the best model dict..")
                best_model = {
                        "model": copy.deepcopy(model.state_dict()),
                        "path": f"./{save_dir}/{epoch:03}_{args.evaluation}_{val_evaluation:4.2%}.pth"
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
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--wdb_on', type=bool, default=False, help='turn on wandb(if you set True)')
    parser.add_argument('--task', type=str, default='default', help='select task you want(default: default) ex: [mask, gender, age]')
    parser.add_argument('--confusion', type=bool, default=False, help='make confusion matrix about each task, logging on wandb')
    # loss 말고 acc, f1 선택할 수 있는 기능
    parser.add_argument('--evaluation', type=str, default='f1', help='set evaluation function (accuracy, f1)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)
    assert not ((args.confusion == True) and (args.task == 'default')), 'confusion and default can\'t exist at the same project'
    if args.wdb_on:
        wandb.init(
            project="Mask image Classification Competition",
            notes="age, focal, val-f1",
            config={
                "img_size": args.resize,
                "loss": args.criterion,
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": args.dataset,
                "augmentation": args.augmentation,
                "epochs": args.epochs,
                "optimizer": args.optimizer,
                "batch_size": args.batch_size
            }
        )
        wandb.run.name = args.name
        wandb.run.save()
        

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
