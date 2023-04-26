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
            print(idx)
            images = images.to(device)
            mask_outs, gender_outs, age_outs = model(images) 

            mask_preds.extend(list(mask_outs.cpu().numpy())) # [0, 0, 0]
            gender_preds.extend(list(gender_outs.cpu().numpy())) # [0]
            age_preds.extend(list(age_outs.cpu().numpy())) # [0, 0, 0]

    mask_preds = np.array(mask_preds)
    age_preds = np.array(age_preds)
    gender_preds = np.array(gender_preds)

    multi_info['mask1'] = mask_preds[:, 0]
    multi_info['mask2'] = mask_preds[:, 1] 
    multi_info['mask3'] = mask_preds[:, 2] 

    multi_info['gender'] = gender_preds[:, 0]

    multi_info['age1'] = age_preds[:, 0]
    multi_info['age2'] = age_preds[:, 1] 
    multi_info['age3'] = age_preds[:, 2] 

    save_path = os.path.join(output_dir, f'output')
    multi_info.to_csv(save_path+'prob_multi.csv', index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs='+', type=int, default=[128, 96], help='resize size for image when you trained (default: [128, 96])')
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
