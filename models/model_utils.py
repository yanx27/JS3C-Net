# *_*coding:utf-8 *_*
"""
Author: Xu Yan
File: model_utils.py
Date: 2020/4/22 13:20
"""
import numpy as np
import torch.nn as nn
import torch
import glob
from spconv.modules import SparseModule
from collections import OrderedDict
import spconv
import models.conv_base as conv_base
import nearest_neighbors as nearest_neighbors

def checkpoint_restore(model, exp_name, use_cuda=True, train_from=0):
    if use_cuda:
        model.cpu()
    epoch = -1
    f = sorted(glob.glob(exp_name + '/model*epoch*' + '.pth'))
    # f = sorted(glob.glob(exp_name + '/model_latest' + '.pth'))
    if len(f) > 0:
        checpoint = f[-1]
        print('Restore from ' + checpoint)
        model.load_state_dict(torch.load(checpoint))
        try:
            epoch = int(checpoint[checpoint.find('epoch') + 5: checpoint.find('.pth')])
        except:
            epoch = 0
    else:
        print('No existing model, starting training from scratch...')

    if use_cuda:
        model.cuda()

    if train_from > 0:
        epoch = train_from

    return epoch


class VoxelPooling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args['Completion']['fuse_k'] > 1:
            self.relation_w = nn.Conv1d(10, args['Segmentation']['m'], 1)

    @staticmethod
    def index_feat(feature, index):
        device = index.device
        N, K = index.shape
        mask = None
        if K > 1:
            group_first = index[:, 0].view((N, 1)).repeat([1, K]).to(device)
            mask = index == 0
            index[mask] = group_first[mask]
        flat_index = index.reshape((N * K,))
        selected_feat = feature[flat_index, ]
        if K > 1:
            selected_feat = selected_feat.reshape((N, K, -1))
        else:
            selected_feat = selected_feat.reshape((N, -1))
        return selected_feat, mask

    @staticmethod
    def relation_position(group_xyz, center_xyz):
        K = group_xyz.shape[1]
        tile_center = center_xyz.unsqueeze(1).repeat([1, K, 1])
        offset = group_xyz - tile_center
        dist = torch.norm(offset, p=None, dim=-1, keepdim=True)
        relation = torch.cat([offset, tile_center, group_xyz, dist], -1)
        return relation

    def forward(self, invoxel_xyz, invoxel_map, src_feat, voxel_center=None):
        device = src_feat.device
        voxel2point_map = invoxel_map[:, :self.args['Completion']['fuse_k']].long()
        features, mask = self.index_feat(src_feat, voxel2point_map)  # [N, K, m]

        if self.args['Completion']['fuse_k'] > 1:
            if self.args['Completion']['pooling_type'] == 'mean':
                features = features.mean(1)
            elif self.args['Completion']['pooling_type'] == 'max':
                features = features.max(1)[0]
            elif self.args['Completion']['pooling_type'] == 'relation':
                '''Voxel relation learning'''
                invoxel_xyz = invoxel_xyz[:, :self.args['Completion']['fuse_k']].to(device)
                N, K, _ = invoxel_xyz.shape
                group_first = invoxel_xyz[:, 0].view((N, 1, 3)).repeat([1, K, 1]).to(device)
                invoxel_xyz[mask, :] = group_first[mask, :]
                relation = self.relation_position(invoxel_xyz, voxel_center.to(device))
                group_w = self.relation_w(relation.permute(0, 2, 1))
                features = features.permute(0, 2, 1)
                features *= group_w
                features = torch.mean(features, 2)

        return features


class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


    def forward(self, seg_pred, seg_gt, seg_w, complt_pred, complt_gt, complt_w, invalid_voxels, sigma):
        '''Loss'''
        '''Segmentation loss'''
        loss_seg = torch.nn.functional.cross_entropy(seg_pred, seg_gt.long(), weight=seg_w)

        '''Completion loss'''
        masks = torch.ones_like(complt_gt, dtype=torch.bool)
        masks[:,:,:,:] = False
        masks[invalid_voxels == 1] = True
        complt_gt[masks] = 255

        loss_complet = 0
        for id, pred in enumerate(complt_pred): # consider multiple head loss computation
            complet_label = complt_gt.long()
            loss_complet += torch.nn.functional.cross_entropy(pred, complet_label, weight=complt_w, ignore_index=255)
        loss_complet /= len(complt_pred)

        if self.args['TRAIN']['uncertainty_loss']:
            factor_seg = 1.0 / (sigma[0]**2)
            factor_complt = 1.0 / (sigma[1]**2)
            loss = factor_seg * loss_seg + \
                   factor_complt * loss_complet +\
                   2 * torch.log(sigma[0]) + \
                   2 * torch.log(sigma[1])
        else:
            loss = self.args['TRAIN']['loss_weight'][0] * loss_seg + self.args['TRAIN']['loss_weight'][1] * loss_complet

        return loss, loss_seg, loss_complet


class interaction_module(nn.Module):
    def __init__(self, args, point_range, voxelsize, k=8, feat_relation=False):
        super().__init__()
        self.feat_relation = feat_relation
        self.point_range = point_range
        self.voxelsize = voxelsize
        self.k = k
        m = args['Segmentation']['m'] if args['Completion']['feeding'] != 'prob' else 19
        if feat_relation:
            self.relation_w = conv_base.Conv1d(in_size=30+m,
                                               out_size=m,
                                               kernel_size=1,
                                               bn=True,
                                               activation=nn.LeakyReLU(0.2))
        else:
            self.relation_w = conv_base.Conv1d(in_size=10,
                                               out_size=m,
                                               kernel_size=1,
                                               bn=True,
                                               activation=nn.LeakyReLU(0.2))
        self.fuse_mlps = conv_base.FC(in_size=m+20,
                                      out_size=20,
                                      bn=True,
                                      activation=nn.LeakyReLU(0.2))

    def forward(self, feat1, feat2):
        '''feat1 is sparsetensor and feat2 is densetensor'''
        device = feat1.features.device
        spatial_shape = feat2.spatial_shape
        indices = feat2.indices

        batch_size = feat1.batch_size
        coord1, features1 = extract_coord_features(feat1.dense().detach())
        coord2, features2 = extract_coord_features(feat2.dense())
        xyz1 = align_pnt(coord1, self.voxelsize, self.point_range) # [0, -25.6, -2, 51.2, 25.6, 4.4]
        xyz2 = align_pnt(coord2, self.voxelsize*2, self.point_range)
        ind = torch.zeros(features2.shape[0], self.k)
        offsets = 0
        offsetq = 0

        for b in range(batch_size):
            support_points = xyz1[xyz1[:, 0] == b][:, 1:].unsqueeze(0).to('cpu').data.numpy() # [N,3]
            query_points = xyz2[xyz2[:, 0] == b][:, 1:].unsqueeze(0).to('cpu').data.numpy() # [M,3]
            indexs = nearest_neighbors.knn_batch(support_points, query_points, self.k, omp=True)
            assert len(indexs[0]) == len(query_points[0])

            num_spoint = support_points.shape[1]
            num_qpoint = query_points.shape[1]
            ind[offsetq:num_qpoint+offsetq] = torch.Tensor(indexs+offsets).squeeze(0)

            offsets += num_spoint
            offsetq += num_qpoint

        ind = ind.long().to(device)
        group, _ = VoxelPooling.index_feat(torch.cat([xyz1[:,1:].float(), features1],1), ind)
        group_xyz, group_features = group[:, :, :3], group[:, :, 3:] # [N, K, 3], [N, K, D]

        relation = VoxelPooling.relation_position(group_xyz, xyz2[:, 1:].float()) # [N, K, 10]
        if self.feat_relation:
            relation = torch.cat([relation, group_features, features2.unsqueeze(1).repeat((1,self.k, 1))], -1)
        group_w = self.relation_w(relation.permute(0,2,1))
        group_features = group_features.permute(0, 2, 1)

        group_features *= group_w
        updated_features = torch.mean(group_features, 2)
        features2 = self.fuse_mlps(torch.cat([updated_features, features2], 1))
        feat = spconv.SparseConvTensor(features2.float(), indices.int(), spatial_shape, batch_size)

        feat.indice_dict = feat2.indice_dict
        feat.grid = feat2.grid

        return feat.dense()


def align_pnt(pnt, voxel_size, pnt_range):
    device = pnt.device
    pnt_range = torch.Tensor(pnt_range[0:3]).to(device)
    pnt[:, 1:] = (pnt[:, 1:].float() + 0.5) * voxel_size + pnt_range

    return pnt.float()


def extract_coord_features(t):
    device = t.device
    channels = int(t.shape[1])
    coords = torch.sum(torch.abs(t), dim=1).nonzero().type(torch.int32).to(device)
    features = t.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]
    features = features.squeeze(1)
    assert len(coords) == len(features)

    return coords, features


def pnt2obj(points, file, rgb=False):
    fout = open('%s.obj' % file, 'w')
    for i in range(points.shape[0]):
        if not rgb:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], 255, 255, 0))
        else:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], points[i, -3]*255, points[i, -2]*255, points[i, -1]*255))
    fout.close()


class PixelShuffle3D(nn.Module):
    """
    3D pixelShuffle
    """
    def __init__(self, upscale_factor):
        """
        :param upscale_factor: int
        """
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)

def check_occupation(complt_input, complt_feat):
    batch_vx = complt_input.to('cpu').data.numpy()
    batch_vx[batch_vx>1] = 1
    complt_feat = torch.sum(torch.abs(complt_feat), dim=1)
    complt_feat[complt_feat!=0] = 1
    complt_feat = complt_feat.squeeze(1).to('cpu').data.numpy()
    print('Alignment rate: ',
          np.sum((batch_vx == complt_feat) & (complt_feat == 1)) /
          len(complt_feat[complt_feat == 1]))

'''Basic component for segmentation spconv'''

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) sparsetensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    batch_size = known_feats.batch_size
    features = known_feats.features
    channel = features.shape[-1]
    ind = torch.zeros(unknown.shape[0], 1)
    offsetq = 0
    offsets = 0

    for b in range(batch_size):
        support_points = known[known[:, 0] == b][:, 1:].unsqueeze(0).to('cpu').data.numpy()  # [N,3]
        query_points = unknown[unknown[:, 0] == b][:, 1:].unsqueeze(0).to('cpu').data.numpy()  # [M,3]
        indexs = nearest_neighbors.knn_batch(support_points, query_points, 1, omp=True)

        num_spoint = support_points.shape[1]
        num_qpoint = query_points.shape[1]
        ind[offsetq:num_qpoint + offsetq] = torch.Tensor(indexs + offsets).squeeze(0)

        offsets += num_spoint
        offsetq += num_qpoint
    ind = ind.long()
    interpolated_feat = features[ind.long()].view(-1,channel)

    return interpolated_feat.contiguous()



class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)

class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, padding=1, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output