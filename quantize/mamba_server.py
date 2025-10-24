import torch
import torch.nn as nn
import torch.distributed as dist
from mamba.configuration import make_split_config
from mamba.mixer import MambaModel
from mamba.weight_split import synchronize_weights
from inference_server import InferenceServer, ThreadLogger

def split_model(full_model: MambaModel, world: int, rank_offset: int = 0) -> list[nn.Module]:
  full_config = full_model.config

  assert full_config.hidden_size % world == 0

  models = []
  if world == 1:
    models.append(full_model.to(f'cuda:{rank_offset}').eval()) # type: ignore
  else:
    for i in range(world):
      config = make_split_config(full_config, world) # type: ignore
      config.rank = i
      config.world_size = world
      model = MambaModel(config)
      for block, full_block in zip(model.layers, full_model.layers):
        block.norm.weight = nn.Parameter(full_block.norm.weight.clone()) # type: ignore
        block.norm.variance_epsilon = full_block.norm.variance_epsilon # type: ignore
        synchronize_weights(full_block.mixer, block.mixer) # type: ignore
      model.norm_f.weight = nn.Parameter(full_model.norm_f.weight.clone())
      model.norm_f.variance_epsilon = full_model.norm_f.variance_epsilon
      models.append(model.eval().to(f'cuda:{i + rank_offset}')) # type: ignore

  print(f'Split models successfully. Will return {len(models)} models')
  return models # type: ignore

def server_factory(method: str, n_gpus: int) -> InferenceServer:
  match method:
    case "DP":
      return MambaDPServer(n_gpus)
    case "TP":
      return MambaTPServer(n_gpus)
    case "HP":
      return MambaHPServer(n_gpus)
    case _:
      raise ValueError(f"Invalid method: {method}")

class MambaDPServer(InferenceServer):
  def load_models(self) -> list[nn.Module]:
    # print("[MambaDPServer] loading model (self.load_models)")
    # full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    # models = []
    # for rank in range(self.n_gpus):
    #   device = torch.device(f'cuda:{rank}')
    #   print(f"[MambaDPServer] load_model, device={device}")
    #   models.append(full_model.to(device=device, non_blocking=False)) # type: ignore
    # return models
    
    full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    # return [full_model.to(f'cuda:{i}') for i in range(self.n_gpus)] # type: ignore
    return [full_model for i in range(self.n_gpus)] # type: ignore
  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    dim0 = input_embeds.size(0)
    world = self.n_gpus
    base_size = dim0 // world
    remainder = dim0 % world

    splits = [base_size + 1] * remainder + [base_size] * (world - remainder)
    split_input_embeds = []
    for i, inp in enumerate(torch.split(input_embeds, splits, dim=0)):
      # split_input_embeds.append(inp.to(f'cuda:{i}'))
      split_input_embeds.append(inp)
    return split_input_embeds
  
  @staticmethod
  def _infer(logger: ThreadLogger, rank: int, models: list[nn.Module], split_input_embeds: torch.Tensor, output: torch.Tensor) -> None:
    logger.info(f"_infer")
    world = len(models)
    batches_per_rank = output.size(0) // world
    model = models[rank].to(f'cuda:{rank}').eval()
    logger.info(f"Model moved to cuda:{rank}")
    #
    inp = split_input_embeds[rank]
    out = model(inputs_embeds=inp.to(f'cuda:{rank}')).last_hidden_state.detach()
    slice_start = rank * batches_per_rank
    slice_end = slice_start + inp.size(0)
    logger.info(f"Slice {slice_start}:{slice_end}")
    output[slice_start:slice_end] = out.cpu()
    
# TODO: Fix another worker spawned (taking minimal amount of memory)
class MambaTPServer(InferenceServer):
  def load_models(self) -> list[nn.Module]:
    full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    return split_model(full_model, self.n_gpus)
  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    split_input_embeds = []
    for i in range(self.n_gpus):
      split_input_embeds.append(input_embeds.to(f'cuda:{i}'))
    return split_input_embeds

  @staticmethod
  def _infer(logger: ThreadLogger, rank: int, models: list[nn.Module], split_input_embeds: torch.Tensor, output: torch.Tensor) -> None:
    device = torch.device(f'cuda:{rank}')
    cleanup = lambda: dist.destroy_process_group() if dist.is_initialized() else None
    dist.init_process_group(backend='nccl', rank=rank, device_id=device)
    model = models[rank]
    try:
      model = models[rank].eval()
      inp = split_input_embeds[rank]
      out = model(inputs_embeds=inp).last_hidden_state.detach()
      if rank == 0:
        output[:] = out.cpu()
    finally:
      cleanup()
  
class MambaHPServer(InferenceServer):
  """
  Hybrid Parallelism Server - combination of Data and Tensor Parallelism
  Data is always split in two (regardless of number of GPUs)
  """
  
  def load_models(self) -> list[nn.Module]:
    full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    assert self.n_gpus >= 4, "HP is only possible on at least 4 GPUs"
    assert self.n_gpus % 2 == 0, "HP is only supported for even number of GPUs"
    return split_model(full_model, self.n_gpus // 2) + split_model(full_model, self.n_gpus // 2, rank_offset=self.n_gpus // 2)
  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    data_split = input_embeds.tensor_split(2, dim=0)
    gpus_per_batch = self.n_gpus // 2
    split_input_embeds = []
    for i in range(gpus_per_batch):
      split_input_embeds.append(data_split[0].to(f'cuda:{i}'))
    for i in range(gpus_per_batch, self.n_gpus):
      split_input_embeds.append(data_split[1].to(f'cuda:{i}'))
    return split_input_embeds
  
  @staticmethod
  def _infer(logger: ThreadLogger, rank: int, models: list[nn.Module], split_input_embeds: torch.Tensor, output: torch.Tensor) -> None:
    world = len(models)
    device = torch.device(f'cuda:{rank}')
    cleanup = lambda: dist.destroy_process_group() if dist.is_initialized() else None
    dist.init_process_group(backend='nccl', rank=rank, device_id=device)
    model = models[rank]
    try:
      model = models[rank].eval()
      inp = split_input_embeds[rank]
      if rank < world // 2:
        model.config.group = dist.new_group(list(range(world // 2))) # type: ignore
      else:
        model.config.group = dist.new_group(list(range(world // 2, world))) # type: ignore
      
      inp = split_input_embeds[rank]
      out = model(inputs_embeds=inp).last_hidden_state.detach()
      if rank == 0:
        output[:world//2] = out.cpu()
      elif rank == world // 2:
        output[world//2:] = out.cpu()
    finally:
      cleanup()
