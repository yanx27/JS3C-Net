# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
# from .utils import toLongTensor
from torch.autograd import Variable


class SparseConvNetTensor(object):
    def __init__(self, features=None, metadata=None, spatial_size=None):
        self.features = features
        self.metadata = metadata
        self.spatial_size = spatial_size

    def get_spatial_locations(self, spatial_size=None):
        "Coordinates and batch index for the active spatial locations"
        if spatial_size is None:
            spatial_size = self.spatial_size
        t = self.metadata.getSpatialLocations(spatial_size)
        return t

    def getLocationsIndexInRef(self, ref):
        "Get spatial locations index in reference metadata"

        idxs = self.metadata.getLocationsIndexInRef(ref.metadata, self.spatial_size)
        return torch.LongTensor(idxs)

    def getSpatialSize(self):
        return self.spatial_size

    def batch_size(self):
        "Batch size"
        t = self.metadata.getBatchSize(self.spatial_size)
        return t

    def extractStructure_(self, ffi, spatial_size, label, subset, kernel_size):
        pass

    def extractStructure(self, label, kernel_size):
        "extract locations according to label and kernel size"
        subset = torch.LongTensor(label.size(0)).zero_()
        kernel_size = toLongTensor(self.metadata.dimension, kernel_size)
        self.extractStructure_(self.metadata.ffi, self.spatial_size, label, subset, kernel_size)
        return subset

    def to(self, device):
        self.features=self.features.to(device)
        return self

    def type(self, t=None):
        if t:
            self.features = self.features.type(t)
            return self
        return self.features.type()

    def cuda(self):
        self.features = self.features.cuda()
        return self

    def cpu(self):
        self.features = self.features.cpu()
        return self

    def detach(self):
        return SparseConvNetTensor(self.features.detach(), self.metadata, self.spatial_size)
    
    @property
    def requires_grad(self):
        return self.features.requires_grad

    def __repr__(self):
        sl = self.get_spatial_locations() if self.metadata else None
        return 'SparseConvNetTensor<<' + \
            'features=' + repr(self.features) + \
            ',features.shape=' + repr(self.features.shape) + \
            ',batch_locations=' + repr(sl) + \
            ',batch_locations.shape=' + repr(sl.shape if self.metadata else None) + \
            ',spatial size=' + repr(self.spatial_size) + \
            '>>'
