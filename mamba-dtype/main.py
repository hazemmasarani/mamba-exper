import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from mixer import *
import pickle as pkl
from time import time
import sys


def setup_environment(world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12345)
    os.environ['WORLD_SIZE'] = str(world_size)
    # dist.init_process_group(backend='nccl', rank=0, world_size=2)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def make_config(config: MambaConfig, world: int):
    return MambaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size // world,
        state_size=config.state_size,
        num_hidden_layers=config.num_hidden_layers,
        layer_norm_epsilon=config.layer_norm_epsilon,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        expand=config.expand,
        conv_kernel=config.conv_kernel,
        use_bias=config.use_bias,
        use_conv_bias=config.use_conv_bias,
        hidden_act=config.hidden_act,
        initializer_range=config.initializer_range,
        residual_in_fp32=config.residual_in_fp32,
        time_step_rank=config.time_step_rank,
        time_step_scale=config.time_step_scale,
        time_step_min=config.time_step_min,
        time_step_max=config.time_step_max,
        time_step_init_scheme=config.time_step_init_scheme,
        time_step_floor=config.time_step_floor,
        rescale_prenorm_residual=config.rescale_prenorm_residual,
        use_cache=config.use_cache,
        use_mambapy=config.use_mambapy,
    )

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

def split_model(full_model: MambaModel):
    world = int(os.environ['WORLD_SIZE'])
    full_config = full_model.config
    
    assert full_config.hidden_size % world == 0

    models = []
    for i in range(world):
        # config = MambaConfig(hidden_size=full_config.hidden_size // world, use_bias=full_config.use_bias, use_conv_bias=full_config.use_conv_bias, use_cache=False, use_mambapy=False)
        config = make_config(full_config, world)
        config.rank = i
        config.world_size = world
        model = MambaModel(config)
        for block, full_block in zip(model.layers, full_model.layers):
            block.norm.weight = nn.Parameter(full_block.norm.weight.clone())
            block.norm.variance_epsilon = full_block.norm.variance_epsilon
            synchronize_weights(full_block.mixer, block.mixer)
        model.norm_f.weight = nn.Parameter(full_model.norm_f.weight.clone())
        model.norm_f.variance_epsilon = full_model.norm_f.variance_epsilon
        models.append(model)

    print(f'Split models successfully. Will return {len(models)} models')
    return models

def run(rank: int, input: torch.Tensor, models: list[MambaModel]):
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(backend='nccl', rank=rank, device_id=device)
    
    try:
        print(f'[Rank {rank}] | Started run')
        model = models[rank].eval().to(f'cuda:{rank}')
        
        start = time()
        # out = model(input.to(device=device))
        output = model(inputs_embeds=input.to(device=device), output_hidden_states=True)
        out = output.last_hidden_state
        # print(f'[Rank {rank}] | Got output', out)
        # dist.all_reduce(out)
        print(f'[Rank {rank}] | Finished distributed run')
        # with open(f'out{rank}.pkl', 'wb') as fp:
        #     pkl.dump(out.detach().cpu(), fp)
        if rank == 0:
            print(f'TP Mixer took {time() - start:.4f}s', out.shape)
            with open('output.pkl', 'wb') as fp:
                pkl.dump(output, fp)

    except Exception as err:
        cleanup()
        raise err
    cleanup()

def main(dtype=torch.float32):
    world_size = 2
    setup_environment(world_size)
    input = torch.tensor(np.random.randint(1, 5e4, size=(1, 16, 2560)), dtype=dtype, device='cpu')
    # config = MambaConfig(hidden_size=768, use_bias=True, use_conv_bias=True)
    # full_model = MambaModel(config)
    full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf")
    # print("Full Model", full_model)
    i = 0
    for layer in full_model.layers:
        print(i, layer)
        i += 1

    sys.exit()

    models = split_model(full_model)

    # mp.set_start_method('spawn', force=True)
    mp.start_processes(run, args=(input, models), nprocs=world_size, join=True, start_method="spawn")
    with open('output.pkl', 'rb') as fp:
        split_model_output: MambaOutput = pkl.load(fp)
        split_model_out = split_model_output.last_hidden_state.cpu()

    full_model = full_model.eval().cuda()
    start = time()
    # full_model_out = full_model(input.cuda()).cpu()
    full_model_output = full_model(inputs_embeds=input.cuda(), output_hidden_states=True)
    full_model_out = full_model_output.last_hidden_state.cpu()
    print(f'Full Mixer took {time() - start:.4f}s')

    # with open('out.pkl', 'wb') as fp:,
    #     pkl.dump(full_model_out.detach().cpu(), fp)

    print((full_model_out == split_model_out).sum() / split_model_out.numel(), torch.isclose(full_model_out, split_model_out, rtol=1e-5, atol=1e-7).sum() / split_model_out.numel())
    # print(out)
    # print((full_model_out.tensor_split(2, dim=0)[0] == out).sum() / out.numel(), torch.isclose(full_model_out.tensor_split(2, dim=0)[0], out, rtol=0, atol=1e-5).sum() / out.numel())
    # print((full_model_out.tensor_split(2, dim=1)[0] == out).sum() / out.numel(), torch.isclose(full_model_out.tensor_split(2, dim=1)[0], out, rtol=0, atol=1e-5).sum() / out.numel())

    # print("\n\n---------------------------------------------------------------------------------------------------")
    # for i, (full_block_out, tp_block_out) in enumerate(zip(split_model_output.hidden_states, full_model_output.hidden_states)):
    #     full_block_out = full_block_out[0].cpu()
    #     tp_block_out = tp_block_out[0].cpu()
    #     if i == 10 or i == 11:
    #         print(full_block_out)
    #     print("Block", i, (full_block_out == tp_block_out).sum() / tp_block_out.numel(), torch.isclose(full_block_out, tp_block_out, rtol=1e-5, atol=1e-6).sum() / tp_block_out.numel())
    # print("---------------------------------------------------------------------------------------------------")

    # #? Just checking how much the output is changed
    # print("\n\n---------------------------------------------------------------------------------------------------")
    # block_out_zero = split_model_output.hidden_states[0][0].cpu()
    # for i, block_out in enumerate(split_model_output.hidden_states):
    #     block_out = block_out[0].cpu()
    #     print("Block", i, (block_out == block_out_zero).sum() / block_out_zero.numel(), torch.isclose(block_out, block_out_zero, rtol=1e-6, atol=1e-8).sum() / block_out_zero.numel())
    # print("---------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()
