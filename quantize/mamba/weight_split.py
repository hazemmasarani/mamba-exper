import torch
import torch.nn as nn
from .mixer import MambaMixer

def synchronize_weights(mixer1: MambaMixer, mixer2: MambaMixer):
    if mixer2.hidden_size > mixer1.hidden_size:
        return synchronize_weights(mixer2, mixer1)
    # From this point, it is fixed that mixer1 is the larger Mixer layer and we are splitting that layer here
    splits = mixer1.hidden_size // mixer2.hidden_size
    index = mixer2.config.rank

    assert mixer1.conv1d.groups % splits == 0

    in_proj_wt_splits = mixer1.in_proj.weight.tensor_split(splits * 2, dim=0)
    in_proj_bias_splits = mixer1.in_proj.bias.tensor_split(splits * 2, dim=0) if mixer1.in_proj.bias is not None else None
    mixer2.in_proj.weight = nn.Parameter(torch.cat((in_proj_wt_splits[index], in_proj_wt_splits[splits+index])))
    mixer2.in_proj.bias = nn.Parameter(torch.cat((in_proj_bias_splits[index], in_proj_bias_splits[splits+index]))) if in_proj_bias_splits else None

    mixer2.conv1d.weight = nn.Parameter(mixer1.conv1d.weight.tensor_split(splits, dim=0)[index].clone())
    mixer2.conv1d.bias = nn.Parameter(mixer1.conv1d.bias.tensor_split(splits, dim=0)[index].clone())
    mixer2.conv1d.groups = mixer1.conv1d.groups // splits

    mixer2.x_proj.weight = nn.Parameter(mixer1.x_proj.weight.tensor_split(splits, dim=-1)[index].clone())

    mixer2.dt_proj.weight = nn.Parameter(mixer1.dt_proj.weight.tensor_split(splits, dim=0)[index].clone())
    mixer2.dt_proj.bias = nn.Parameter(mixer1.dt_proj.bias.tensor_split(splits, dim=0)[index].clone())

    mixer2.out_proj.weight = nn.Parameter(mixer1.out_proj.weight.tensor_split(splits, dim=-1)[index].clone())
    if index == 0 and mixer1.out_proj.bias is not None:
        mixer2.out_proj.bias = nn.Parameter(mixer1.out_proj.bias.clone())
    else:
        mixer2.out_proj.bias = None

    mixer2.A_log = nn.Parameter(mixer1.A_log.tensor_split(splits, dim=0)[index].clone())
    mixer2.D = nn.Parameter(mixer1.D.tensor_split(splits, dim=0)[index].clone())