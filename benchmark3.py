import argparse
import torch
import pickle
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Compute next-token probabilities from embeddings")
    parser.add_argument("--input", type=str, required=True, help="Path to input pickle file (embeddings)")
    parser.add_argument("--output", type=str, required=True, help="Path to output pickle file (probabilities)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load embeddings
    with open(args.input, "rb") as f:
        hidden_states_all = pickle.load(f)  # shape [num_samples, batch, seq_len, hidden_size]

    print(f"Loaded embeddings: {hidden_states_all.shape}")

    # 2. Load model + LM head
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").eval()
    lm_head = model.get_output_embeddings()

    # 3. Device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_head = lm_head.to(device)
    dtype = torch.float32

    # 4. Process each sample, take last token, store in list
    all_probs = []

    with torch.no_grad():
        for i, hidden_states in enumerate(hidden_states_all):
            hidden_states = torch.as_tensor(hidden_states, dtype=dtype, device=device)

            # Take only last token hidden state for next-token prediction
            last_hidden = hidden_states[:, -1, :]  # shape [batch, hidden_size]

            # Compute logits → softmax
            logits = lm_head(last_hidden)          # shape [batch, vocab_size]
            probs = torch.softmax(logits, dim=-1).cpu()  # move back to CPU

            all_probs.append(probs)
            print(f"✅ Processed sample {i+1}/{len(hidden_states_all)} (shape {probs.shape})")

    # 5. Stack all into single tensor: [num_samples * batch, vocab_size]
    final_probs = torch.cat(all_probs, dim=0)
    print(f"\n✅ Final probability tensor shape: {final_probs.shape}")

    # 6. Save single tensor
    with open(args.output, "wb") as f:
        pickle.dump(final_probs, f)

    print(f"💾 All next-token probabilities saved to {args.output}")

if __name__ == "__main__":
    main()
