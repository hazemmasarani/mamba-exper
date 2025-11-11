import torch
import datasets
from transformers import AutoTokenizer
import pickle as pkl

def tokenize_and_prepare(dataset_name="wikitext", subset="wikitext-2-raw-v1", model_name="state-spaces/mamba-2.8b-hf"):
    # Load dataset
    dataset = datasets.load_dataset(dataset_name, subset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize each text sample
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=2560,
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Convert to PyTorch tensors
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Example: Get tensors from validation split
    input_ids = tokenized_dataset["validation"]["input_ids"]

    return input_ids  # torch.Tensor of token IDs

# Example usage
embeddings = tokenize_and_prepare()

BATCH_SIZE = 2
SEQ_LEN = 512
EMBEDDING_SIZE = 2560

# Function to create rolling sequences
def rolling_sequences(tensor, seq_len):
    sequences = []
    for i in range(len(tensor) - seq_len + 1):
        sequences.append(tensor[i:i+seq_len])
    return torch.stack(sequences)  # shape [num_sequences, seq_len, embedding_size]

def batching_sequences(tensor, batch_size):
    batches = []
    for i in range(batch_size, len(tensor), batch_size):
        batches.append(tensor[i-batch_size:i])
    return torch.stack(batches)  # shape [num_batches, batch_size, seq_len, embedding_size]

def rolling_sequences_output(tensor, seq_len):
    sequences = []
    for i in range(seq_len, len(tensor)):
        sequences.append(tensor[i:i+1])
    return torch.stack(sequences)  # shape [num_sequences, seq_len, embedding_size]

def batching_sequences_output(tensor, batch_size):
    batches = []
    for i in range(0, len(tensor), batch_size):
        batches.append(tensor[i:i+batch_size])
    return torch.stack(batches)  # shape [num_batches, batch_size, seq_len, embedding_size]

input_ids = rolling_sequences(embeddings, SEQ_LEN)

input_ids = batching_sequences(input_ids, BATCH_SIZE)

output_ids = rolling_sequences_output(embeddings, SEQ_LEN)

output_ids = batching_sequences_output(output_ids, BATCH_SIZE)

print(input_ids.shape)
print(output_ids.shape)
with open("tokenizer.pkl", "wb") as f:
    pkl.dump({"input_ids": input_ids, "output_ids": output_ids}, f)