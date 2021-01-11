import torch
from torch.autograd import Variable
import sparseconvnet as scn

def abstract(dimension, input, prediction, kernel_size = 1):
    prediction = prediction.features.data
    _, predicted = prediction.max(1)
    if predicted.sum() == 0:
        print('dangerous! no predicted structure')
        predicted[:] = 1
    structure = input.extractStructure(predicted.cpu(), kernel_size).nonzero().view(-1)
    input_locations = input.get_spatial_locations()
    input_features = input.features
    output_locations = torch.index_select(input_locations, 0, structure)

    # structure = Variable(structure, requires_grad=False).cuda()
    output_features = torch.index_select(input_features, 0, structure)

    output = scn.InputBatch(dimension, input.getSpatialSize())
    output.setLocations(output_locations, torch.Tensor(output_features.data.shape))
    output.features = output_features #Preserve autograd continuity
    return output