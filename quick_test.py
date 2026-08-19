import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from app import load_or_init_model, generate_text

print("Loading model...")
model, stoi, itos, cfg = load_or_init_model()
print(f"Model vocab size: {cfg.vocab_size}")
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join(itos.get(i, '�') for i in l)
prompt = "Hello"
print(f"Prompt: {prompt}")
out = generate_text(prompt, max_new_tokens=10, temperature=0.8, top_k=40)
print(f"Output: {out}")
print("Test succeeded.")