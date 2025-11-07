import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from abc import ABC, abstractmethod
import traceback
import time

class OOMException(Exception):
  """Raised when inference exceeds available memory."""
  pass

class ThreadLogger:
  name: str
  error_queue: mp.Queue
  
  def __init__(self, name: str, error_queue: mp.Queue, verbose: bool = False):
    self.name = name
    self.error_queue = error_queue
    self.verbose = verbose
    
  def info(self, msg: str):
    print(f"[INFO {self.name}] {msg}")
  
  def debug(self, msg: str):
    if self.verbose:
      print(f"[DEBUG {self.name}] {msg}")
  
  def error(self, err: str, tb: str | None = None):
    print(f"[ERR {self.name}] {err}")
    self.error_queue.put((err, tb))

class InferenceServer(ABC):
  n_gpus: int
  models: list[nn.Module]

  def __init__(self, n_gpus: int, quantize_dtype: str = "float32"):
    # torch.set_num_threads(1)
    if mp.get_start_method(allow_none=True) != 'spawn':
      mp.set_start_method('spawn', force=True)
    self.n_gpus = n_gpus
    self.models = []
    self.quantize_dtype = quantize_dtype
    print(f"{self.__class__.__name__} initialized with {n_gpus} GPUs")

  @abstractmethod
  def load_models(self) -> list[nn.Module]:
    raise NotImplementedError("Replace with real model loading call")

  @abstractmethod  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    raise NotImplementedError("Replace with real splitting call")
  
  # This method is now re-entrant, so no harm in calling it multiple times (use force to load models again)
  def prepare(self, force: bool = False):
    if force or not self.models:
      self.models = self.load_models()

  @staticmethod
  @abstractmethod
  def _infer(logger: ThreadLogger, rank: int, world: int, model: nn.Module, split_input: torch.Tensor, output: torch.Tensor) -> None:
    """Thread worker"""
    raise NotImplementedError("Replace with per-thread inference call")

  @classmethod
  def _infer_wrapper(cls, error_queue, rank, world, *args, **kwargs):
    """
    Wrapper that calls the subclass's _infer method and captures exceptions.
    """

    logger = ThreadLogger(f"{cls.__name__} Rank {rank}", error_queue, verbose=True)
    logger.debug("in _infer_wrapper")
    try:
      os.environ['MASTER_ADDR'] = 'localhost'
      os.environ['MASTER_PORT'] = str(12345)
      os.environ['WORLD_SIZE'] = str(world)
      os.environ['RANK'] = str(rank)
      start = time.perf_counter()
      cls._infer(logger, rank, world, *args, **kwargs)
      logger.info(f"finished _infer in {(time.perf_counter() - start)*1000:.0f} ms")
      logger.debug("destroying process group")
      if dist.is_initialized():
        dist.destroy_process_group()
      logger.debug("destroyed process group")
      error_queue.put(None)
    except Exception as err:
      tb = traceback.format_exc()
      logger.error(str(err), tb)

  def infer(self, input_embeds: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input_embeds, dtype=torch.float32)
    split_input_embeds = self.split_input(input_embeds)
    processes = []
    error_queues = []  # Error queue for each process (needed for clean output and to control error reporting)

    for i in range(self.n_gpus):
      error_queue = mp.Queue()
      error_queues.append(error_queue)
      start = time.perf_counter()
      p = mp.Process(target=self._infer_wrapper, args=(error_queue, i, len(self.models), self.models[i], split_input_embeds[i], output))
      p.start()
      # print(f"Process {i} started in {time.perf_counter() - start} seconds")
      processes.append(p)
      
    for p in processes:
      p.join()

    # pool = mp.Pool(self.n_gpus).
    # # pool.

    for i, eq in enumerate(error_queues):
      err = eq.get()
      if err is not None:
        msg, tb = err
        if dist.is_initialized():
          dist.destroy_process_group()
        # torch.cuda.empty_cache()
        if "CUDA out of memory" in str(msg):
          raise OOMException()
        raise Exception(f"Child process {i} failed:\n{msg}\nTraceback:\n{tb}")

    # torch.cuda.empty_cache()
    return output
