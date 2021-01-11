# *_*coding:utf-8 *_*
"""
Author: Xu Yan
File: test_kitti_ssc.py
Date: 2020/4/9 14:40
"""
import sys
import os
import json
import spconv
import time
import yaml
import torch

import importlib
import argparse
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from models import model_utils
import sparseconvnet as scn
from datetime import datetime
import kitti_dataset

'''Inference'''
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, default='log/JS3C-Net-kitti/', help='Experiment root')
    parser.add_argument('--dataset', type=str, default='valid', help='[valid/test]')
    return parser.parse_args()

args = parse_args()

print('Load Model...')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_path = args.log_dir

output_dir = model_path + '/dump/'
if not os.path.exists(output_dir): os.mkdir(output_dir)
output_dir = output_dir + 'completion'
if not os.path.exists(output_dir): os.mkdir(output_dir)
submit_dir = output_dir + '/submit_' + args.dataset + datetime.now().strftime('%Y_%m_%d')
if not os.path.exists(submit_dir): os.mkdir(submit_dir)

use_cuda = torch.cuda.is_available()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(model_path)

with open(os.path.join(model_path, 'args.txt'), 'r') as f:
    config = json.load(f)
config['GENERAL']['debug'] = False

nClasses = 20

config_file = os.path.join('opt', 'semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
valid_labels = np.zeros((20), dtype=np.int32)
learning_map_inv = kitti_config['learning_map_inv']
for key,value in learning_map_inv.items():
    valid_labels[key] = value

seg_head = importlib.import_module('models.' + config['Segmentation']['model_name'])
seg_model = seg_head.get_model

complet_head = importlib.import_module('models.' + config['Completion']['model_name'])
complet_model = complet_head.get_model

class J3SC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = config
        self.seg_head = seg_model(config)
        self.complet_head = complet_model(config)
        self.voxelpool = model_utils.VoxelPooling(config)
        self.seg_sigmas_sq = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.complet_sigmas_sq = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)

    def forward(self, x):
        seg_inputs, complet_inputs, _ = x

        '''Segmentation Head'''
        seg_output, feat = self.seg_head(seg_inputs)
        torch.cuda.empty_cache()

        '''Completion Head'''
        coords = complet_inputs['complet_coords']
        coords = coords[:, [0, 3, 2, 1]]

        if self.args['DATA']['dataset'] == 'SemanticKITTI':
            coords[:, 3] += 1  # TODO SemanticKITTI will generate [256,256,31]
        elif self.args['DATA']['dataset'] == 'SemanticPOSS':
            coords[:, 3][coords[:, 3] > 31] = 31

        if self.args['Completion']['feeding'] == 'both':
            feeding = torch.cat([seg_output, feat], 1)
        elif self.args['Completion']['feeding'] == 'feat':
            feeding = feat
        else:
            feeding = seg_output
        features = self.voxelpool(invoxel_xyz=complet_inputs['complet_invoxel_features'][:, :, :-1],
                                  invoxel_map=complet_inputs['complet_invoxel_features'][:, :, -1].long(),
                                  src_feat=feeding,
                                  voxel_center=complet_inputs['voxel_centers'])
        if self.args['Completion']['no_fuse_feat']:
            features[...] = 1
            features = features.detach()

        batch_complet = spconv.SparseConvTensor(features.float(), coords.int(), self.args['Completion']['full_scale'],
                                                self.args['TRAIN']['batch_size'])
        batch_complet = dataset.sparse_tensor_augmentation(batch_complet, complet_inputs['state'])

        if self.args['GENERAL']['debug']:
            model_utils.check_occupation(complet_inputs['complet_input'], batch_complet.dense())

        complet_output = self.complet_head(batch_complet)
        torch.cuda.empty_cache()

        return seg_output, complet_output

classifier = J3SC_Net()
if use_cuda:
    classifier = classifier.cuda()
classifier = classifier.eval()

scn.checkpoint_restore(classifier, model_path, use_cuda)
print('#classifer parameters %d' % sum([x.nelement() for x in classifier.parameters()]))

dataset = importlib.import_module('kitti_dataset')
input_data = kitti_dataset.get_dataset(config, split = args.dataset)
data_loader = torch.utils.data.DataLoader(
    input_data,
    batch_size=1,
    collate_fn=seg_head.Merge,
    num_workers=config['TRAIN']['train_workers'],
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
)
num_sample = len(data_loader)
print("# files: {}".format(num_sample))

with torch.no_grad():
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        start = time.time()
        sequence, filename = batch[2][0]
        os.makedirs(os.path.join(submit_dir, 'sequences', sequence, 'predictions'), exist_ok=True)
        full_save_dir = os.path.join(submit_dir, 'sequences', sequence, 'predictions', filename + '.label')

        if os.path.exists(full_save_dir):
            print('%s already exsist...' % (full_save_dir))
            continue

        seg_pred, complet_pred= classifier(batch)

        pred_choice_complet = complet_pred[-1].data.max(1)[1]
        pred = pred_choice_complet.cpu().long().data.numpy()

        # make lookup table for mapping
        maxkey = max(learning_map_inv.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())

        pred = pred.astype(np.uint32)
        pred = pred.reshape((-1))
        upper_half = pred >> 16  # get upper half for instances
        lower_half = pred & 0xFFFF  # get lower half for semantics
        lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
        pred = (upper_half << 16) + lower_half  # reconstruct full label
        pred = pred.astype(np.uint32)

        # Save
        final_preds = pred.astype(np.uint16)
        final_preds.tofile(full_save_dir)
