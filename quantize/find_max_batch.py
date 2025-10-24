import torch
import csv
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from inference_server import OOMException
from mamba_server import server_factory

# Parameters
SEQUENCE_LENGTHS = [128]
N_GPUS_LIST = [2]
EMBEDDING_SIZE = 2560
METHODS = ["DP", "TP", "HP"]            # Array of any of "DP", "TP", "HP"
MODEL_SIZE = 12 * 1024**3
GPU_MEM = 40 * 1024**3
DTYPE = "float32"        # "float32", "float16", "bfloat16", "int8"
OUTPUT_FILE = "mamba_results/max_batches_40G.csv"

# INTERNAL PARAMS
DTYPE_SIZES = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
METHOD_FACTORS = {"DP": None, "HP": 0.5, "TP": 1.0}

CSV_COLUMNS = [
"timestamp", "method", "n_gpus", "seq_len",
"embedding_size", "dtype", "max_batch"
]

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
        return set()
    df = pd.read_csv(p, usecols=["method","n_gpus","seq_len","embedding_size","dtype"])
    return set(zip(df["method"], df["n_gpus"], df["seq_len"], df["embedding_size"], df["dtype"]))

def append_result(path: str, row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

# ---- Batch Upper bound calculation ----
def calc_batch_ub(seq_len: int, n_gpus: int, method: str) -> int:
    dtype_bytes = DTYPE_SIZES[DTYPE]
    size_per_batch = seq_len * EMBEDDING_SIZE * dtype_bytes

    per_gpu_avail = max(0, GPU_MEM - MODEL_SIZE)
    if method == "DP":
        effective_mem = per_gpu_avail
    else:
        factor = METHOD_FACTORS[method]
        effective_mem = int(n_gpus * per_gpu_avail * factor)

    if size_per_batch == 0:
        return 0
    return int((effective_mem * 0.01) / size_per_batch)

def find_max_batch(server, seq_len: int, method: str, n_gpus: int) -> int:
    high = max(1, calc_batch_ub(seq_len, n_gpus, method))
    low, best = 1, 0
    print(f"Finding max batch for method={method}, n_gpus={n_gpus} | UB={high}")
    while low <= high:
        mid = (low + high) // 2
        input_embeds = torch.randint(1, int(5e4), size=(mid, seq_len, EMBEDDING_SIZE), dtype=torch.long, device='cpu')
        try:
            print(f"Trying with input embedding size {input_embeds.shape}")
            out = server.infer(input_embeds)
            print(f"Got output of shape {out.shape}")
            best = mid
            low = mid + 1
        except OOMException:
            print(f"OOM error for {method} with {n_gpus} GPUs and shape {input_embeds.shape}")
            high = mid - 1
    return best

def main():
  ensure_output_file(OUTPUT_FILE)
  existing = load_existing_results(OUTPUT_FILE)
  print(existing)

  for method in METHODS:
    for n_gpus in N_GPUS_LIST:
        if method == "HP" and n_gpus < 4:
            continue
        if n_gpus > torch.cuda.device_count():
            continue
        server = server_factory(method, n_gpus)
        server.prepare()
        for seq_len in SEQUENCE_LENGTHS:
            print(f"Finding max batch for {method}, seq_len={seq_len}, n_gpus={n_gpus}")
            key = (method, n_gpus, seq_len, EMBEDDING_SIZE, DTYPE)
            if key in existing:
                continue            
            max_batch = find_max_batch(server, seq_len, method, n_gpus)
            print(f"Returned max_batch={max_batch}")
            row = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "n_gpus": n_gpus,
                "seq_len": seq_len,
                "embedding_size": EMBEDDING_SIZE,
                "dtype": DTYPE,
                "max_batch": max_batch,
            }
            append_result(OUTPUT_FILE, row)
            existing.add(key)

if __name__ == "__main__":
    main()
