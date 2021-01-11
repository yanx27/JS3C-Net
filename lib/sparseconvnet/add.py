# by zjh, ILC, 2018.01.03

import torch
from torch.autograd import Variable
from torch.nn import Module
import sparseconvnet as scn
from .utils import add_feature_planes
from .sparseConvNetTensor import SparseConvNetTensor

def Add_fun(input1, input2):
    '''input1 and input2 has the same positions'''
    output = add_feature_planes([input1, input2])
    return output

def Add2_fun(input1, input2):
    '''output position is the same as input2'''
    output = SparseConvNetTensor()
    output.metadata = input2.metadata
    output.spatial_size = input2.spatial_size
    input1_features = torch.zeros(input2.features.size()).cuda()
    idxs = input2.getLocationsIndexInRef(input1)
    hit = (idxs != -1).nonzero().view(-1)
    input1_features[hit] = input1.features[idxs[hit]]
    output.features = input1_features + input2.features
    return output
