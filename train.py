import torch
from Config import configure_model
from architecture import build_model
from Tokenizer import build_tokenizer  # BPE trainer (optional)
from train_setup import train_step
import math, os

def get_char_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

def main():
    # ----- hyper-params (edit as you wish) -----
    class CFG: pass
    cfg = CFG()
    cfg = configure_model(cfg)          # gets n_layer, n_head, etc.
    cfg.vocab_size = 50257              # GPT-2 vocab size – change if you use a different tokenizer
    cfg.block_size = 256
    cfg.batch_size = 60
    cfg.max_steps = 5000                # total training steps
    cfg.learning_rate = 6e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    # ----- data -----
    if not os.path.isfile("input.txt"):
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write("Hello world! This is a tiny demo corpus.\n")
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    # Use character vocab for simplicity; replace with BPE if you like
    stoi, itos = get_char_vocab(text)
    cfg.vocab_size = len(stoi)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)

    # ----- model -----
    model = build_model(cfg).to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

    # ----- optimizer -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ----- training loop -----
    for step in range(cfg.max_steps):
        xb, yb = get_batch(data, cfg.block_size, cfg.batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    # ----- checkpoint -----
    ckpt = {
        "model_state_dict": model.state_dict(),
        "cfg": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")},
        "stoi": stoi,
        "itos": itos,
    }
    torch.save(ckpt, "tinylm.ckpt")
    print("✅ Checkpoint saved to tinylm.ckpt")

if __name__ == "__main__":
    main()
