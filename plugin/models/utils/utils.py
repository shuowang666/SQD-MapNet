import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule, ModuleList
import math
from mmcv.cnn import Linear


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor,
                               num_feats,
                               temperature=10000,
                               scale=2 * math.pi):
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                        dim=3).flatten(2)  # TODO: .view()
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                        dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                            dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                            dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            pos_tensor.size(-1)))
    # import ipdb; ipdb.set_trace()
    return pos


def SinePositionalEncoding(mask,
                            num_feats,
                            temperature=10000,
                            normalize=False,
                            scale=2 * math.pi,
                            eps=1e-6,
                            offset=0.,):
        mask = mask.cumsum(0)
        # import ipdb; ipdb.set_trace()
        if normalize:
            mask = mask / (mask[-1:] + eps) * scale
        dim_t = torch.arange(num_feats, dtype=torch.float32, device=mask.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_feats)
        pos = mask[:, None] / dim_t
        pos = torch.stack((pos[:, 0::2].sin(), pos[:, 1::2].cos()), dim=2).view(-1, num_feats)
        return pos


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN) with relu."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



