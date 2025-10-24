import torch
import csv
import os
import pandas as pd
import pickle as pkl
from datetime import datetime
from pathlib import Path
from inference_server import InferenceServer, OOMException
from mamba_server import server_factory
from time import perf_counter

# Parameters
# List of (batch_size, seq_len) tuples
BENCHMARK_INPUTS = [(17, 128)]
N_GPUS_LIST = [4]
EMBEDDING_SIZE = 2560
METHODS = ["TP"]            # Array of any of "DP", "TP", "HP"
DTYPE = "float32"        # "float32", "float16", "bfloat16", "int8"
COMM_DTYPE = "float16"    # "float32", "float16"
OUTPUT_FILE = "mamba_results/max_batches_40G_benchmark.csv"
SANITY_CHECK_FILE = "sample_4.64.2560.float32.pkl"
SKIP_EXISTING = False
SKIP_SANITY_CHECK = False
RUNS_PER_INPUT = 1

# INTERNAL PARAMS
CSV_COLUMNS = [
  "timestamp", "method", "n_gpus", "batch_size", "seq_len",
  "embedding_size", "dtype", "comm_dtype", "time"
]
# Uniquely identify each run
KEY = ["method", "n_gpus", "batch_size", "seq_len", "embedding_size", "dtype", "comm_dtype"]

def ensure_output_file(path: str):
  p = Path(path)
  os.makedirs(p.parent, exist_ok=True)
  if not p.exists():
    with p.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
      writer.writeheader()

def load_existing_results(path: str):
  p = Path(path)
  if not p.exists():
    return list()
  df = pd.read_csv(p, usecols=KEY)
  return list(zip(*[df[col] for col in KEY]))

def append_result(path: str, row: dict):
  with open(path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writerow(row)
    f.flush()
    os.fsync(f.fileno())

def sanity_check(server: InferenceServer):
  print("======================== Sanity Check ========================")
  with open(SANITY_CHECK_FILE, "rb") as f:
    sinp, sout = pkl.load(f)
  start = perf_counter()
  out = server.infer(sinp).detach().cpu() # type: ignore
  time = perf_counter() - start
  close_mat = torch.isclose(out, sout, rtol=1e-2)
  ratio = close_mat.sum() / close_mat.numel()
  if ratio < 0.99:
    raise Exception(f"Sanity check failed: only {ratio:.2%} elements are close (took {time:.2f} seconds)")
  print(f"Sanity check passed: {ratio:.2%} (took {time:.2f} seconds)")

def main():
  ensure_output_file(OUTPUT_FILE)
  existing = load_existing_results(OUTPUT_FILE)

  for method in METHODS:
    for n_gpus in N_GPUS_LIST:
      if method == "HP" and n_gpus < 4:
        continue
      if n_gpus > torch.cuda.device_count():
        continue
      server = server_factory(method, n_gpus)
      server.prepare()
      if not SKIP_SANITY_CHECK:
        sanity_check(server)
      for batch_size, seq_len in BENCHMARK_INPUTS:
        key = (method, n_gpus, batch_size, seq_len, EMBEDDING_SIZE, DTYPE, COMM_DTYPE)
        runs = existing.count(key)
        if SKIP_EXISTING and runs >= RUNS_PER_INPUT:
          continue
        for _ in range(RUNS_PER_INPUT - runs):
          inp = torch.randint(1, int(5e4), size=(batch_size, seq_len, EMBEDDING_SIZE), dtype=torch.long, device='cpu')
          try:
            start = perf_counter()
            server.infer(inp)
            time = perf_counter() - start
          except OOMException:
            time = float("inf")
            print(f"OOM for {method}, {n_gpus} GPUs and shape {inp.shape}")
          except Exception as e:
            print(f"Exception {e} for {method} with {n_gpus} GPUs and shape {inp.shape}")
            time = -1
          row = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "n_gpus": n_gpus,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "embedding_size": EMBEDDING_SIZE,
            "dtype": DTYPE,
            "comm_dtype": COMM_DTYPE,
            "time": time,
          }
          append_result(OUTPUT_FILE, row)

if __name__ == "__main__":
  main()
