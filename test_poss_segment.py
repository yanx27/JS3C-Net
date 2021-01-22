#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: test_poss_segment.py
@time: 2020/9/22 14:08
'''
import sys
import os
import torch
import time
import math
import yaml
import json
import importlib
import argparse

import torch.nn as nn
from tqdm import tqdm
import numpy as np

from models import model_utils
from datetime import datetime
import sparseconvnet as scn
from utils import laserscan
from utils.np_ioueval import iouEval

'''Inference'''
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, default='JS3C-Net-POSS', help='Experiment root')
    parser.add_argument('--num_votes', type=int, default=10, help='Aggregate segmentation scores with voting [default: 10]')
    return parser.parse_args()

args = parse_args()

print('Load Model...')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_path = 'log/'+args.log_dir
val_reps = args.num_votes

output_dir = model_path + '/dump/'
if not os.path.exists(output_dir): os.mkdir(output_dir)
output_dir = output_dir + 'segmentation_poss'
if not os.path.exists(output_dir): os.mkdir(output_dir)
submit_dir = output_dir + '/submit_' + datetime.now().strftime('%Y_%m_%d')
if not os.path.exists(submit_dir): os.mkdir(submit_dir)

use_cuda = torch.cuda.is_available()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(model_path)
with open(model_path+'/args.txt', 'r') as f:
    config = json.load(f)
print(config)

seg_head = importlib.import_module('models.' + config['Segmentation']['model_name'])
seg_model = seg_head.get_model
complet_head = importlib.import_module('models.' + config['Completion']['model_name'])
complet_model = complet_head.get_model

class J3SC_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seg_head = seg_model(config)
        self.complet_head = complet_model(config)
        self.voxelpool = model_utils.VoxelPooling(config)
        self.seg_sigmas_sq = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.complet_sigmas_sq = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)

    def forward(self, x):
        seg_output, _ = self.seg_head(x)

        return seg_output

classifier = J3SC_Net(config)
print(classifier)

if use_cuda:
    classifier = classifier.cuda()
classifier = classifier.eval()

training_epoch = scn.checkpoint_restore(classifier, model_path, use_cuda)
print('#classifer parameters %d' % sum([x.nelement() for x in classifier.parameters()]))

'''Load Dataset'''
config_file = os.path.join('opt/SemanticPOSS.yaml')
poss_config = yaml.safe_load(open(config_file, 'r'))
scan = laserscan.SemLaserScan(nclasses=11, sem_color_dict=poss_config['color_map'])
sequences = poss_config['split']['valid']

label_to_names = {0: 'people', 1: 'rider', 2: 'car', 3: 'trunk', 4: 'plants', 5: 'traffic-sign', 6: 'pole', 7: 'building',
                  8: 'fence', 9: 'bike', 10: 'ground'}

points = []
for sequence in sequences:
    sequence = '{0:02d}'.format(int(sequence))
    points_path = os.path.join(config['GENERAL']['dataset_dir'], 'sequences', sequence, 'velodyne')
    seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(points_path) if pn.endswith('.bin')]
    seq_points_name.sort()
    points.extend(seq_points_name)

valid_labels = np.zeros((11), dtype=np.int32)
learning_map_inv = poss_config['learning_map_inv']
for key,value in learning_map_inv.items():
    if key > 0:
        valid_labels[key-1] = value

def process_data(points_name):
    locs=[]
    feats=[]
    point_ids=[]
    for idx,i in enumerate(range(val_reps)):
        scan.open_scan(points_name)
        label_name = points_name.replace('bin', 'label').replace('velodyne', 'labels')
        scan.open_label(label_name)
        label = scan.sem_label
        label = label.astype(np.int32)

        remissions = scan.remissions
        coords = scan.points
        point_num = len(coords)
        if config['Segmentation']['use_coords']:
            feature = np.concatenate([coords, remissions.reshape(-1, 1)], 1)
        else:
            feature = remissions.reshape(-1, 1)
        coords = np.ascontiguousarray(coords - coords.mean(0))
        m = np.eye(3)
        m[0][0] *= np.random.randint(0,2)*2-1
        m *= config['Segmentation']['scale']
        theta = np.random.rand()*2*math.pi
        m = np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        coords = np.matmul(coords,m)+config['Segmentation']['full_scale'][1]/2+np.random.uniform(-2,2,3)
        m = coords.min(0)
        M = coords.max(0)
        offset =- m+np.clip(config['Segmentation']['full_scale'][1]-M+m-0.001,0,None)*np.random.rand(3)+np.clip(config['Segmentation']['full_scale'][1]-M+m+0.001,None,0)*np.random.rand(3)
        coords += offset
        idxs = (coords.min(1)>=0)*(coords.max(1)<config['Segmentation']['full_scale'][1])
        coords = coords[idxs]
        feature = feature[idxs]
        coords = torch.Tensor(coords).long()
        locs.append(torch.cat([coords,torch.LongTensor(coords.shape[0],1).fill_(idx)],1))
        feats.append(torch.Tensor(feature))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]))
    locs = torch.cat(locs,0)
    feats = torch.cat(feats,0)
    point_ids = torch.cat(point_ids,0)
    labels = torch.Tensor(label)

    return {'seg_coords': locs,
            'seg_features': feats,
            'y': labels.long(),
            'point_ids': point_ids,
            'length':point_num}

classifier.eval()
with torch.no_grad():
    NUM_CLASS_SEG = config['DATA']['classes_seg']
    evaluator = iouEval(NUM_CLASS_SEG, [])

    remapdict = poss_config["learning_map"]
    # make lookup table for mapping
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut = remap_lut - 1
    remap_lut[remap_lut == -1] = -100

    for idx, filename in tqdm(enumerate(points),total=len(points)):
        components = filename.split('/')
        sequence = components[-3]
        points_name = components[-1]
        label_name = points_name.replace('bin', 'label')
        full_save_dir = os.path.join(submit_dir, 'sequences', sequence, 'predictions')
        os.makedirs(full_save_dir, exist_ok=True)
        full_label_name = os.path.join(full_save_dir, label_name)

        start = time.time()
        batch = process_data(filename)
        store = torch.zeros(batch['length'], 11)

        predictions = classifier(batch)
        store.index_add_(0, batch['point_ids'], predictions.cpu())
        pred = store.max(1)[1].numpy().astype(int)
        label = remap_lut[batch['y'].long().data.numpy().astype(int)]
        pred = pred[label != -100]
        label = label[label != -100]

        evaluator.addBatch(pred, label)

    _, class_jaccard = evaluator.getIoU()
    m_jaccard = class_jaccard.mean()
    for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=label_to_names[i], jacc=jacc * 100))
    print('Eval point avg class IoU: %f' % (m_jaccard * 100))













