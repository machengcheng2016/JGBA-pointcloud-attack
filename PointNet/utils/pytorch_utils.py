import torch
import torch.nn as nn
from torch.autograd import Variable

def to_var(x, requires_grad=False, cuda=True):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if cuda:
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def requires_grad_(model:nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def get_source_layers(model_name, model):
    if model_name == 'PointNet':
        # exclude relu, feat, dropout
        layer_list = list(map(lambda name: (name, model._modules.get(name)), ['fc1', 'bn1', 'fc2', 'bn2', 'fc3']))
        return list(enumerate(layer_list))

    elif model_name == 'DenseNet121':
        # exclude relu, convs, dropout
        layer_list = list(map(lambda name: (name, model._modules.get(name)), ['linear1', 'bn6', 'linear2', 'bn7', 'linear3']))
        return list(enumerate(layer_list))

    else:
        # model is not supported
        assert False


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input