# Tiny LLM Playground

A **minimal, educational transformer** (≈2.7 M parameters) that you can
*inspect*, *train*, and *generate* with via a Gradio web interface.

## 📁 Repository layout
```
Tiny_llm/
├─ Config.py          # sets default hyper-parameters (layers, heads, …)
├─ Tokenizer.py       # BPE trainer (optional)
├─ Embeddings.py
├─ Attention.py
├─ feed_forward.py
├─ normalization.py
├─ optimizer.py
├─ LR_scheduler.py
├─ architecture.py    # model + generate() method
├─ train_setup.py     # single training step
├─ app.py             # Gradio launcher
├─ train.py           # optional standalone training script
├─ requirements.txt   # pip packages
└─ README.md
```

## 🚀 Getting started (local)

1. **Clone / download** the repo and `cd` into it.
2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the demo**
   ```bash
   python app.py
   ```
5. Open the URL that appears (usually `http://127.0.0.1:7860`).

## 🎮 Using the UI

| Tab | What you can do |
| --- | --- |
| **Generate** | Type a prompt, adjust temperature, top-k, and max new tokens, click Generate. |
| **Train** | Upload a `.txt` file (any language, any size). Set Epochs, Batch size, Learning rate and click Start Training. The model will be saved as `tinylm.ckpt` and re-loaded automatically for generation. |
| **Info** | Shows the current configuration (layers, heads, embedding size, vocab size, …). Change the numbers in `Config.py` and restart the app to try a different architecture. |

> **Note:** The model starts with random weights. Without training it will produce random-looking tokens. Even a few minutes of training on a modest corpus (e.g., a few MB of plain text) will give you visibly more coherent continuations.

## 🐳 Running in Docker (optional)

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
ENV PORT=7860
CMD ["python", "app.py"]
```

Build & run:
```bash
docker build -t tiny-llm .
docker run -p 7860:7860 tiny-llm
```

## 🌐 Deploying to Hugging-Face Spaces (free)

1. Create a new Space (choose Gradio as the SDK).
2. Upload the whole repository (or push via Git).
3. HF Spaces automatically runs `app.py` – make sure `requirements.txt` is at the root.
4. Your space will be reachable at `https://<your-username>.hf.space`.

## 🛠️ Extending the demo

- Replace the character-level vocab with the BPE merges from `Tokenizer.py` (load merges and build a `stoi`/`itos` dictionary).
- Add more hyper-parameter sliders (e.g., `n_layer`, `n_head`, `n_embd`) – just read them from `Config.py` before rebuilding the model in the Generate or Train callbacks.
- Integrate logging / TensorBoard – plug a `torch.utils.tensorboard.SummaryWriter` into the training loop.
