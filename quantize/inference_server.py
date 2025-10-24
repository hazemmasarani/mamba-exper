import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from abc import ABC, abstractmethod
import traceback

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
    print(f"[DEBUG {self.name}] {msg}")
  
  def error(self, err: str, tb: str | None = None):
    self.error_queue.put((err, tb))

class InferenceServer(ABC):
  n_gpus: int
  models: list[nn.Module]

  def __init__(self, n_gpus: int):
    if not mp.get_start_method(allow_none=True):
      mp.set_start_method('spawn')
    self.n_gpus = n_gpus
    self.models = []
    print(f"{self.__class__.__name__} initialized with {n_gpus} GPUs")

  def setup_env_vars(self):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12345)
    os.environ['WORLD_SIZE'] = str(self.n_gpus)
    
  @abstractmethod
  def load_models(self) -> list[nn.Module]:
    raise NotImplementedError("Replace with real model loading call")

  @abstractmethod  
  def split_input(self, input_embeds: torch.Tensor) -> list[torch.Tensor]:
    raise NotImplementedError("Replace with real splitting call")
  
  def prepare(self):
    self.setup_env_vars()
    self.models = self.load_models()

  @staticmethod
  @abstractmethod
  def _infer(logger: ThreadLogger, rank: int, input_embeds: torch.Tensor, models: list[nn.Module]) -> torch.Tensor:
    """Thread worker"""
    raise NotImplementedError("Replace with per-thread inference call")

  @classmethod
  def _infer_wrapper(cls, error_queue, rank, *args, **kwargs):
    """
    Wrapper that calls the subclass's _infer method and captures exceptions.
    """

    logger = ThreadLogger(f"{cls.__name__} Rank {rank}", error_queue)
    try:
      cls._infer(logger, rank, *args, **kwargs)
      error_queue.put(None)
    except Exception as e:
      tb = traceback.format_exc()
      logger.error(str(e), tb)

  def infer(self, input_embeds: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input_embeds, dtype=torch.float32)
    split_input_embeds = self.split_input(input_embeds)
    processes = []
    error_queues = []  # Error queue for each process

    for i in range(self.n_gpus):
      error_queue = mp.Queue()
      error_queues.append(error_queue)
      p = mp.Process(target=self._infer_wrapper, args=(error_queue, i, self.models, split_input_embeds, output))
      p.start()
      processes.append(p)
    
    for p in processes:
      p.join()
    
    for i, eq in enumerate(error_queues):
      err = eq.get()
      if err is not None:
        if "CUDA out of memory" in str(err):
          raise OOMException()
        msg, tb = err
        raise Exception(f"Child process {i} failed:\n{msg}\nTraceback:\n{tb}")

    return output
