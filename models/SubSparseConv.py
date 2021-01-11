#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: SubSparseConv.py
@time: 2020/4/12 22:46
'''

import sparseconvnet as scn
import torch.nn as nn
from models import conv_base
import torch

def get_model(config):
    return Unet(config)

class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config['Segmentation']['m']
        input_dim = 4 if config['Segmentation']['use_coords'] else 1
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(3, config['Segmentation']['full_scale'][1], mode=4)).add(
           scn.SubmanifoldConvolution(3, input_dim, m, 3, False)).add(
               scn.UNet(dimension=3,
                        reps=config['Segmentation']['block_reps'],
                        nPlanes=[m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m],
                        residual_blocks=config['Segmentation']['block_residual'],
                        groups=config['Segmentation']['seg_groups']
                        )).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(3))
        self.linear = nn.Linear(m, self.config['DATA']['classes_seg'])
        if self.config['Completion']['interaction']:
            self.shape_embedding = conv_base.Conv1d(m, m, kernel_size=1, bn=True, activation=nn.LeakyReLU(0.2))


    def forward(self, x):
        batch_x = [x['seg_coords'], x['seg_features'].cuda()]
        x = self.sparseModel(batch_x)
        if self.config['Completion']['interaction']:
            feat = self.shape_embedding(x.unsqueeze(0).permute(0,2,1))
            feat = feat.squeeze(0).permute(1,0)
            x = feat + x
            x = self.linear(x)
        else:
            feat = x
            x = self.linear(feat)

        return x, feat


def Merge(tbl):
    seg_coords = []
    seg_features = []
    seg_labels = []
    complet_coords = []
    complet_invalid = []
    voxel_centers = []
    complet_invoxel_features = []
    complet_labels = []
    filenames = []
    offset = 0
    input_vx = []
    stats = []
    for idx, example in enumerate(tbl):
        filename, completion_collection, aliment_collection, segmentation_collection = example
        '''File Name'''
        filenames.append(filename)

        '''Segmentation'''
        seg_coord = segmentation_collection['coords']
        seg_coords.append(torch.cat([seg_coord, torch.LongTensor(seg_coord.shape[0], 1).fill_(idx)], 1))
        seg_labels.append(segmentation_collection['label'])
        seg_features.append(segmentation_collection['feature'])

        '''Completion'''
        complet_coord = aliment_collection['coords']
        complet_coords.append(torch.cat([torch.Tensor(complet_coord.shape[0], 1).fill_(idx), torch.Tensor(complet_coord)], 1))

        input_vx.append(completion_collection['input'])
        complet_labels.append(completion_collection['label'])
        complet_invalid.append(completion_collection['invalid'])
        stats.append(completion_collection['stat'])

        voxel_centers.append(torch.Tensor(aliment_collection['voxel_centers']))
        complet_invoxel_feature = aliment_collection['voxels']
        complet_invoxel_feature[:, :, -1] += offset  # voxel-to-point mapping in the last column
        offset += seg_coord.shape[0]
        complet_invoxel_features.append(torch.Tensor(complet_invoxel_feature))

    seg_inputs = {'seg_coords': torch.cat(seg_coords, 0),
                  'seg_labels': torch.cat(seg_labels, 0),
                  'seg_features': torch.cat(seg_features, 0)
                  }

    complet_inputs = {'complet_coords': torch.cat(complet_coords, 0),
                      'complet_input': torch.cat(input_vx, 0),
                      'voxel_centers': torch.cat(voxel_centers, 0),
                      'complet_invalid': torch.cat(complet_invalid, 0),
                      'complet_labels': torch.cat(complet_labels, 0),
                      'state': stats,
                      'complet_invoxel_features': torch.cat(complet_invoxel_features, 0)
                      }

    return seg_inputs, complet_inputs, filenames