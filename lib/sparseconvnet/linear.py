"""
Function to applies a linear transformation to SparseConvNetTensor: :math:`y = Ax + b`.
Minic http://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
"""
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter
import sparseconvnet as scn
from .sparseConvNetTensor import SparseConvNetTensor
import time

class Linear(Module):
    def __init__(self, dimension, in_nPlanes, out_nPlanes, bias=True):
        super(Linear, self).__init__()
        self.dimension = dimension
        self.in_nPlanes = in_nPlanes
        self.out_nPlanes = out_nPlanes
        self.weight = Parameter(torch.Tensor(out_nPlanes, in_nPlanes))
        if bias:
            self.bias = Parameter(torch.Tensor(out_nPlanes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # output_features = F.linear(input.features, self.weight, self.bias)
        # output = scn.InputBatch(self.dimension, input.getSpatialSize())
        # output.setLocations(input.getSpatialLocations(), torch.Tensor(output_features.data.shape))
        # output.features = output_features
        output = SparseConvNetTensor()
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        output.features = F.linear(input.features, self.weight, self.bias)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_nPlanes=' + str(self.in_nPlanes) \
            + ', out_nPlanes=' + str(self.out_nPlanes) + ')'

    def input_spatial_size(self, out_size):
        return out_size
