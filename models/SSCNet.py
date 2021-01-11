# *_*coding:utf-8 *_*
"""
Author: Jiantao Gao
File: complt_sscnet.py
Date: 2020/4/27 17:46
"""
import torch
import torch.nn as nn
from models import model_utils
import spconv

def get_model(config):
    return SSCNet(config)


class SSCNet_Decoder(nn.Module):
    def __init__(self, input_dim, nPlanes, classes):
        super().__init__()
        # Block 1
        self.b1_conv1=nn.Sequential(nn.Conv3d(input_dim, 16, 7, 2, padding=3), nn.BatchNorm3d(16),nn.ReLU())
        self.b1_conv2=nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]),nn.ReLU())
        self.b1_conv3=nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[0], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[0]),nn.ReLU())
        self.b1_res=nn.Sequential(nn.Conv3d(16, nPlanes[0], 3, 1,padding=1), nn.BatchNorm3d(nPlanes[0]),nn.ReLU())
        self.pool1=nn.Sequential(nn.MaxPool3d(2, 2))

        # Block 2
        self.b2_conv1=nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_conv2=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())
        self.b2_res=nn.Sequential(nn.Conv3d(nPlanes[0], nPlanes[1], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[1]),nn.ReLU())

        # Block 3
        self.b3_conv1=nn.Sequential(nn.Conv3d(nPlanes[1], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),nn.ReLU())
        self.b3_conv2=nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[2], 3, 1, padding=1), nn.BatchNorm3d(nPlanes[2]),nn.ReLU())

        # Block 4
        self.b4_conv1=nn.Sequential(nn.Conv3d(nPlanes[2], nPlanes[3], 3, 1, dilation=2, padding=2), nn.BatchNorm3d(nPlanes[3]),nn.ReLU())
        self.b4_conv2=nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[3], 3, 1, dilation=2, padding=2), nn.BatchNorm3d(nPlanes[3]),nn.ReLU())

        # Block 5
        self.b5_conv1=nn.Sequential(nn.Conv3d(nPlanes[3], nPlanes[4], 3, 1, dilation=2, padding=2), nn.BatchNorm3d(nPlanes[4]),nn.ReLU())
        self.b5_conv2=nn.Sequential(nn.Conv3d(nPlanes[4], nPlanes[4], 3, 1, dilation=2, padding=2), nn.BatchNorm3d(nPlanes[4]),nn.ReLU())
        
        # Prediction
        self.pre_conv1=nn.Sequential(nn.Conv3d(nPlanes[2]+nPlanes[3]+nPlanes[4], int((nPlanes[2]+nPlanes[3]+nPlanes[4])/3*2), 1, 1),\
                                     nn.BatchNorm3d(int((nPlanes[2]+nPlanes[3]+nPlanes[4])/3*2)),nn.ReLU())
        self.pre_conv2=nn.Sequential(nn.Conv3d(int((nPlanes[2]+nPlanes[3]+nPlanes[4])/3*2), classes, 1, 1))

    def forward(self, x):
        # Block 1
        x = self.b1_conv1(x)
        res_x = self.b1_res(x)
        x = self.b1_conv2(x)
        x = self.b1_conv3(x)
        x = x + res_x

        # Block 2
        res_x = self.b2_res(x)
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = x +res_x

        # Block 3
        b3_x1 = self.b3_conv1(x)
        b3_x2 = self.b3_conv2(b3_x1)
        b3_x = b3_x1 + b3_x2

        # Block 4
        b4_x1 = self.b4_conv1(b3_x)
        b4_x2 = self.b4_conv2(b4_x1)
        b4_x = b4_x1 +b4_x2

        # Block 5
        b5_x1 = self.b5_conv1(b4_x)
        b5_x2 = self.b5_conv2(b5_x1)
        b5_x = b5_x1 + b5_x2

        # Concat b3,b4,b5
        x = torch.cat((b3_x, b4_x, b5_x),dim=1)

        # Prediction
        x = self.pre_conv1(x)
        x = self.pre_conv2(x)
        return x

class SSCNet(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        classes = args['DATA']['classes_completion']
        m = args['Completion']['m']
        if args['Completion']['feeding'] == 'feat':
            input_dim = args['Segmentation']['m']
        elif args['Completion']['feeding'] == 'both':
            input_dim = args['Segmentation']['m'] + args['DATA']['classes_seg']
        else:
            input_dim = args['DATA']['classes_seg']
        self.Decoder = SSCNet_Decoder(input_dim=input_dim, nPlanes=[m, m, m, m, m], classes=classes)
        self.upsample = nn.Sequential(nn.Conv3d(in_channels=classes, out_channels=classes * 8, kernel_size=1, stride=1),
                                      nn.BatchNorm3d(classes * 8), nn.ReLU(), model_utils.PixelShuffle3D(upscale_factor=2))
        if args['Completion']['interaction']:
            self.interaction_module = model_utils.interaction_module(args,
                                                                     self.args['Completion']['point_cloud_range'],
                                                                     self.args['Completion']['voxel_size'],
                                                                     self.args['Completion']['search_k'],
                                                                     feat_relation=args['Completion']['feat_relation'])

    def forward(self, feat):
        x = feat.dense()
        x = self.Decoder(x)
        if self.args['Completion']['interaction']:
            coord, features = model_utils.extract_coord_features(x)
            if self.args['Completion']['feeding'] == 'both':
                feat.features = feat.features[:, self.args['DATA']['classes_seg']:]
            x = spconv.SparseConvTensor(features=features.float(),
                                        indices=coord.int(),
                                        spatial_shape=[int(s/2) for s in self.args['Completion']['full_scale']],
                                        batch_size=self.args['TRAIN']['batch_size'])
            x = self.interaction_module(feat, x)
        x = self.upsample(x)

        return [x]




