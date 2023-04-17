import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import numpy as np

class MyError(Exception):
    def __str__(self):
        return "too many .pth file in directory"
    
def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model+"_Model")
    
    model = model_cls()

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    #model_path = os.path.join(saved_model, 'best.pth')
    model_list = [file for file in os.listdir(model_dir) if file.endswith(".pth")]
    if len(model_list) < 1:
        raise FileNotFoundError
    elif len(model_list) > 1:
        raise MyError
    model_path = saved_model + '/' + model_list[0]
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    multi_info = info.drop('ans', axis=1)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    mask_preds = []
    gender_preds = []
    age_preds = [] 
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            mask_outs, gender_outs, age_outs = model(images) 
            mask_pred = torch.argmax(mask_outs, dim=-1)
            age_pred = torch.argmax(age_outs, dim=-1)
            gender_pred = torch.where(gender_outs <= torch.tensor(0.5), torch.tensor([0]).to(device), torch.tensor([1]).to(device)).squeeze()
            
            pred = 6 * mask_pred + 3 * gender_pred + age_pred
            
            preds.extend(pred.cpu().numpy())
            mask_preds.extend(mask_pred.cpu().numpy())
            gender_preds.extend(gender_pred.cpu().numpy())
            age_preds.extend(age_pred.cpu().numpy())

    info['ans'] = preds
    multi_info['mask'] = mask_preds 
    multi_info['gender'] = gender_preds
    multi_info['age'] = age_preds 
    save_path = os.path.join(output_dir, f'output')
    info.to_csv(save_path+'_default.csv', index=False)
    multi_info.to_csv(save_path+'_multi.csv', index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=[128, 96], help='resize size for image when you trained (default: [128, 96])')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
