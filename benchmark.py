import os
import argparse
import torch
import pickle as pkl

from inference_server import OOMException
from mamba_server import server_factory
from time import perf_counter

# Parameters


# TODO: Modify according to new benchmark inputs
# List of (method, n_gpus) tuples
BENCHMARK_INPUTS = [("TP", 2)]
TOKENIZER_FILE = "tokenizer.pkl"

CHECKPOINT_FILE = "checkpoint.pkl"
OUTPUTS_FILE = "outputs.pkl"

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference script with resume support")

    parser.add_argument("--checkpoint", type=str, default="checkpoint.pkl",
                        help="Path to checkpoint pickle file")
    parser.add_argument("--outputs", type=str, default="outputs.pkl",
                        help="Path to output pickle file")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.pkl",
                        help="Path to tokenizer pickle file")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type to use (e.g., float32, float16)")

    return parser.parse_args()

def save_checkpoint(index):
    """Save the current index to checkpoint file."""
    with open(CHECKPOINT_FILE, "wb") as f:
        pkl.dump(index, f)

def load_checkpoint():
    """Load the last processed index, or start from -1 if none."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            return pkl.load(f)
    return -1

def append_output(new_output):
  """Append a tensor [2, 1, 2560] to a growing [N, 2, 1, 2560] tensor in the pickle file."""

  if os.path.exists(OUTPUTS_FILE):
      with open(OUTPUTS_FILE, "rb") as f:
          all_outputs = pkl.load(f)  # shape: [k, 2, 1, 2560]
      # ✅ Concatenate along dim=0 to make shape [k+1, 2, 1, 2560]
      all_outputs = torch.cat([all_outputs, new_output.unsqueeze(0)], dim=0)
  else:
      # ✅ Initialize the tensor with shape [1, 2, 1, 2560]
      all_outputs = new_output.unsqueeze(0)

  with open(OUTPUTS_FILE, "wb") as f:
      pkl.dump(all_outputs, f)

def load_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
        return data["input_ids"], data["output_ids"]

def main():
  global BENCHMARK_INPUTS, CHECKPOINT_FILE, OUTPUTS_FILE, TOKENIZER_FILE
  args = parse_args()
  CHECKPOINT_FILE = args.checkpoint
  OUTPUTS_FILE = args.outputs
  TOKENIZER_FILE = args.tokenizer
  quantize_dtype = args.dtype

  print(f"Quantize dtype: {quantize_dtype}")
  
  # Load dataset and prepare one batch
  # tokenized_val = tokenize_and_prepare("wikitext", "wikitext-2-raw-v1")
  # val_loader = batch_data(tokenized_val, batch_size=BENCHMARK_INPUTS[0][2])
  input_ids, _ = load_tokenizer(TOKENIZER_FILE)

  for method, n_gpus in sorted(BENCHMARK_INPUTS):
    server = server_factory(method, n_gpus, quantize_dtype=quantize_dtype)
    server.prepare()
    inp = input_ids
    try:
      time = 0.0
      start_index = load_checkpoint()
      for i in range(start_index + 1, len(input_ids)):
        chunk = input_ids[i]           # shape: [4, 1024, 2560]
        start = perf_counter()
        out = server.infer(chunk)
        time = perf_counter() - start
        print(f"Time for chunk {i}: {time:.2f} seconds")
        append_output(out.detach())
        save_checkpoint(i)
    except OOMException:
      time = float("inf")
      print(f"OOM for {method}, {n_gpus} GPUs and shape {inp.shape}")
    except Exception as e:
      print(f"Exception {e} for {method} with {n_gpus} GPUs and shape inp.shape")
      time = -1

if __name__ == "__main__":
  main()
