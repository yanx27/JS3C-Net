#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: SubSpconv.py
@time: 2020/4/12 22:46
'''

import torch.nn as nn
from models import conv_base
from utils import config
import numpy as np
import functools
import spconv
import torch
from models.model_utils import UBlock, ResidualBlock, VGGBlock
from lib.pointgroup_ops.functions import pointgroup_ops

args = config.cfg

def get_model(config):
    return Unet(config)

class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config['Segmentation']['m']
        input_dim = 4 if config['Segmentation']['use_coords'] else 1
        block_residual = config['Segmentation']['block_residual']
        block_reps = config['Segmentation']['block_reps']

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_dim, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.backbone_net = UBlock([m, 2*m, 3*m, 4*m, 5*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        self.linear = nn.Linear(m, self.config['DATA']['classes_seg'])

        if self.config['Completion']['interaction']:
            self.shape_embedding = conv_base.Conv1d(m, m, kernel_size=1, bn=True, activation=nn.LeakyReLU(0.2))

    def forward(self, data_dict):
        x = self.point_voxelization(data_dict)
        x = self.input_conv(x)
        x = self.backbone_net(x)
        x = self.output_layer(x)
        x = x.features[data_dict['p2v_map'].long()]  # (N, F)

        if self.config['Completion']['interaction']:
            feat = self.shape_embedding(x.unsqueeze(0).permute(0,2,1))
            feat = feat.squeeze(0).permute(1,0)
            x = feat + x
            x = self.linear(x)
        else:
            feat = x
            x = self.linear(feat)

        return x, feat

    def point_voxelization(self, data_dict):
        '''Transfering batch point clouds to a sparse tensor

        Args:
            data_dict: dict

        Returns:
            input：SparseTensor

        '''
        feats = data_dict['seg_features'].cuda()
        v2p_map = data_dict['v2p_map'].cuda()
        voxel_locs = data_dict['voxel_locs'].int().cuda()
        spatial_shape = data_dict['spatial_shape']

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, self.config['Segmentation']['mode'])  # (M, C), float, cuda
        input = spconv.SparseConvTensor(voxel_feats, voxel_locs, spatial_shape, self.config['TRAIN']['batch_size'])

        return input

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
        seg_coords.append(torch.cat([torch.LongTensor(seg_coord.shape[0], 1).fill_(idx), seg_coord.long()], 1))
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

    seg_coords = torch.cat(seg_coords, 0)
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(seg_coords.contiguous(), args['TRAIN']['batch_size'], args['Segmentation']['mode'])
    spatial_shape = np.clip((seg_coords.max(0)[0][1:] + 1).numpy(), args['Segmentation']['full_scale'][0], None)  # long (3)

    seg_inputs = {'seg_coords': seg_coords,
                  'seg_labels': torch.cat(seg_labels, 0),
                  'seg_features': torch.cat(seg_features, 0),
                  'voxel_locs': voxel_locs,
                  'p2v_map': p2v_map,
                  'v2p_map': v2p_map,
                  'spatial_shape': spatial_shape,
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