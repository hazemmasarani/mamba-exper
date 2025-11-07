import torch
from mamba_server import server_factory

# Parameters
method = "TP"
n_gpus = 4
batch_size = 2
seq_len = 512
EMBEDDING_SIZE = 2560
quantize_dtype = "float32"

def main():

  server = server_factory(method, n_gpus, quantize_dtype=quantize_dtype)
  server.prepare()
  inp = torch.randint(1, int(5e4), size=(batch_size, seq_len, EMBEDDING_SIZE), dtype=torch.long, device='cpu')
  out = server.infer(inp)
  print(f"Output shape: {out.shape}")

if __name__ == "__main__":
  main()
