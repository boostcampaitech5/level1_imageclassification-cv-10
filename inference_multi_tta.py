import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import numpy as np
from torchvision.transforms import ColorJitter


class MyError(Exception):
    def __str__(self):
        return "too many .pth file in directory"
    
def load_model( model_path, device):
    model_cls = getattr(import_module("model"), args.model+"_Model")
    
    model = model_cls()

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    #model_path = os.path.join(saved_model, 'best.pth')

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    model_list = [file for file in os.listdir(model_dir) if file.endswith(".pth")]

    img_root = os.path.join(data_dir, 'images')
    info_path = '/opt/ml/input/data/eval/info.csv'
    info = pd.read_csv(info_path)
    multi_info = info.drop('ans', axis=1)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    num_workers = multiprocessing.cpu_count() // 2
    if num_workers ==0:
        num_workers = 1
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
 
    cj = ColorJitter(brightness=(0.2, 1),hue=(-0.3, 0.3))
    oof_mask = None
    oof_age = None
    oof_gender = None
    print(f'models: {model_list}')
    for m_idx,model_name in enumerate(model_list): 
        preds = []
        mask_preds = []
        gender_preds = []
        age_preds = [] 
        model_path = model_dir + '/' + model_name
        model = load_model(model_path, device).to(device)
        model.eval()
        with torch.no_grad():
            for idx, images in enumerate(loader):

                images = images.to(device)
                mask_outs, gender_outs, age_outs = model(images)
                mask_flip_outs, gender_flip_outs, age_flip_outs = model(torch.flip(images, dims=(-1,)))
                mask_cj_outs, gender_cj_outs, age_cj_outs = model(cj(images))

                tmp_mask = mask_outs+mask_flip_outs+mask_cj_outs
                tmp_age = age_outs+age_flip_outs+age_cj_outs
                tmp_gender = (gender_outs+gender_flip_outs+gender_cj_outs)/3

                mask_preds.extend(tmp_mask.cpu().numpy())
                age_preds.extend(tmp_age.cpu().numpy())
                gender_preds.extend(tmp_gender.cpu().numpy())
            fold_mask = torch.tensor(mask_preds)
            fold_age = torch.tensor(age_preds)
            fold_gender = torch.tensor(gender_preds)
            # print(fold_mask)
        print('fold shape',fold_mask.shape)
        if m_idx==0:
            oof_mask = fold_mask
            oof_age = fold_age
            oof_gender = (fold_gender)/len(model_list)
        else:
            oof_mask += fold_mask
            oof_age += fold_age
            oof_gender += (fold_gender)/len(model_list)
        print('oof shape',oof_mask.shape)


    mask_pred = torch.argmax(oof_mask, dim=-1)
    age_pred = torch.argmax(oof_age, dim=-1)
    gender_pred = torch.where(oof_gender <= torch.tensor(0.5), torch.tensor([0]), torch.tensor([1])).squeeze()

    pred = 6 * mask_pred + 3 * gender_pred + age_pred
            


    info['ans'] = pred
    multi_info['mask'] = mask_pred
    multi_info['gender'] = gender_pred
    multi_info['age'] = age_pred
    save_path = os.path.join(output_dir, f'output')
    info.to_csv(save_path+'_default.csv', index=False)
    multi_info.to_csv(save_path+'_multi.csv', index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=int, default=[224, 224], help='resize size for image when you trained (default: [128, 96])')
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
