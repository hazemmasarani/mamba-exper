import queue
import torch
import torch.nn as nn
import torch.distributed as dist
from mamba.configuration import make_split_config
from mamba.mixer import MambaModel
from mamba.weight_split import synchronize_weights
from inference_server import InferenceServer, ThreadLogger, OOMException
import copy

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
  print("Models config quantize size: ", models[0].config.quantize_dtype)
  print(f'Split models successfully. Will return {len(models)} models')
  return models # type: ignore

def server_factory(method: str, n_gpus: int, quantize_dtype: str = "float32") -> InferenceServer:
  match method:
    case "DP":
      return MambaDPServer(n_gpus, quantize_dtype)
    case "TP":
      return MambaTPServer(n_gpus, quantize_dtype)
    case "HP":
      return MambaHPServer(n_gpus, quantize_dtype)
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

    full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf")
    full_model.config.quantize_dtype = self.quantize_dtype
    def _load(rank: int) -> nn.Module:
      if rank == self.n_gpus - 1:
        model = full_model
      else:
        model = copy.deepcopy(full_model)
      return model.to(f'cuda:{rank}').eval() # type: ignore
    return [_load(i) for i in range(self.n_gpus)]
  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    dim0 = input_embeds.size(0)
    world = self.n_gpus
    base_size = dim0 // world
    remainder = dim0 % world

    splits = [base_size + 1] * remainder + [base_size] * (world - remainder)
    split_input_embeds = []
    for i, inp in enumerate(torch.split(input_embeds, splits, dim=0)):
      split_input_embeds.append(inp.to(f'cuda:{i}'))
      
    return split_input_embeds
  
  @staticmethod
  def _infer(logger: ThreadLogger, rank: int, world: int, model: nn.Module, split_input: torch.Tensor, output: torch.Tensor) -> None:
    batches_per_rank = output.size(0) // world
    #
    out = model(inputs_embeds=split_input).last_hidden_state.detach()
    slice_start = rank * batches_per_rank
    slice_end = slice_start + split_input.size(0)
    logger.debug(f"Slice {slice_start}:{slice_end}")
    output[slice_start:slice_end] = out.cpu()

class MambaTPServer(InferenceServer):
  def infer(self, input_embeds: torch.Tensor) -> torch.Tensor:
    if self.n_gpus != 1:
      return super().infer(input_embeds)

    output = torch.empty_like(input_embeds, dtype=torch.float32)
    split_input_embeds = self.split_input(input_embeds)
    error_queue = queue.SimpleQueue()

    self._infer_wrapper(error_queue, 0, 1, self.models[0], split_input_embeds[0], output)
    err = error_queue.get()
    if err is not None:
      msg, tb = err
      if dist.is_initialized():
        dist.destroy_process_group()
      if "CUDA out of memory" in str(msg):
        raise OOMException()
      raise Exception(f"Child process 0 failed:\n{msg}\nTraceback:\n{tb}")

    return output

  def load_models(self) -> list[nn.Module]:
    full_model = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    full_model.config.quantize_dtype = self.quantize_dtype
    return split_model(full_model, self.n_gpus)
  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    split_input_embeds = []
    for i in range(self.n_gpus):
      split_input_embeds.append(input_embeds.to(f'cuda:{i}'))
    return split_input_embeds

  @staticmethod
  def _infer(logger: ThreadLogger, rank: int, world: int, model: nn.Module, split_input: torch.Tensor, output: torch.Tensor) -> None:
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(backend='nccl', rank=rank, device_id=device)
    out = model(inputs_embeds=split_input).last_hidden_state.detach()
    if rank == 0:
      output[:] = out.cpu()
  
  def get_output_embeddings(self):
    # Assuming all GPU shards share the same head
    return self.models[0].get_output_embeddings()
  
class MambaHPServer(InferenceServer):
  """
  Hybrid Parallelism Server - combination of Data and Tensor Parallelism
  Data is always split in two (regardless of number of GPUs)
  """
  
  def load_models(self) -> list[nn.Module]:
    assert self.n_gpus >= 4, "HP is only possible on at least 4 GPUs"
    assert self.n_gpus % 2 == 0, "HP is only supported for even number of GPUs"

    full_model1 = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    full_model1.config.quantize_dtype = self.quantize_dtype
    full_model2 = MambaModel.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    full_model2.config.quantize_dtype = self.quantize_dtype
    return split_model(full_model1, self.n_gpus // 2) + split_model(full_model2, self.n_gpus // 2, rank_offset=self.n_gpus // 2)
  
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
  def _infer(logger: ThreadLogger, rank: int, world: int, model: nn.Module, split_input: torch.Tensor, output: torch.Tensor) -> None:
    device = torch.device(f'cuda:{rank}')
    dist.init_process_group(backend='nccl', rank=rank, device_id=device)
    # index at which we split into 2 TP groups
    split_index = world // 2
    if rank < split_index:
      model.config.group = dist.new_group(list(range(split_index))) # type: ignore
    else:
      model.config.group = dist.new_group(list(range(split_index, world))) # type: ignore

    out = model(inputs_embeds=split_input).last_hidden_state.detach()
    # DP from this batch
    batch = output.size(0) // 2
    if rank == 0:
      output[:batch] = out.cpu()
    elif rank == split_index:
      output[batch:] = out.cpu()
