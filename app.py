import os
import torch
import string
import gradio as gr
from architecture import build_model
from Config import configure_model

# ----------------------------------------------------------------------
# Helper: build a character-level tokenizer with ASCII coverage.
# ----------------------------------------------------------------------
UNK_CHAR = ""

def get_char_vocab(text):
    # Standard printable ASCII + any unique chars in dataset
    chars = sorted(list(set(string.printable).union(set(text))))
    if UNK_CHAR not in chars:
        chars = [UNK_CHAR] + chars
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def safe_encode(s, stoi):
    unk_idx = stoi.get(UNK_CHAR, 0)
    return [stoi.get(c, unk_idx) for c in s]

def safe_decode(l, itos):
    return "".join(itos.get(i, UNK_CHAR) for i in l)

# ----------------------------------------------------------------------
# Built-in Datasets (Short, clean tags ideal for character-level models)
# ----------------------------------------------------------------------
DEFAULT_SHAKESPEARE = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep—
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life;
"""

DEFAULT_DIALOGUE = """User: Hi
Bot: Hello! How are you doing today?

User: Hello
Bot: Hey there! What are you working on?

User: How are you?
Bot: I'm doing well! How about you?

User: I'm good.
Bot: Glad to hear it! What can I help you with?

User: I'm building a tiny language model.
Bot: That's awesome! Are you training nanoGPT from scratch?

User: Yeah, nanoGPT.
Bot: Nice! Is it learning context-aware conversations?

User: What is Python?
Bot: Python is a popular programming language used for web apps and AI.

User: Explain Python simply.
Bot: Python lets you give instructions to a computer using clear, simple code.

User: I don't understand.
Bot: Think of Python like a recipe that tells a computer what steps to follow!
"""

# ----------------------------------------------------------------------
# Load (or create) model + vocab checkpoint
# ----------------------------------------------------------------------
def load_or_init_model(ckpt_path="tinylm.ckpt", default_corpus="input.txt"):
    if os.path.isfile(ckpt_path):
        print(f"[INFO] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        class CFG: pass
        cfg = CFG()
        for k, v in ckpt["cfg"].items():
            setattr(cfg, k, v)

        cfg.vocab_size = len(ckpt["itos"])
        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        stoi, itos = ckpt["stoi"], ckpt["itos"]
        return model, stoi, itos, cfg

    print("[INFO] No checkpoint found – initializing fresh model.")
    if not os.path.isfile(default_corpus):
        with open(default_corpus, "w", encoding="utf-8") as f:
            f.write(DEFAULT_DIALOGUE)
    with open(default_corpus, "r", encoding="utf-8") as f:
        text = f.read()

    stoi, itos = get_char_vocab(text)
    cfg = configure_model(type("Cfg", (), {})())
    cfg.vocab_size = len(stoi)
    cfg.block_size = 256
    cfg.max_steps = 1000
    model = build_model(cfg)
    return model, stoi, itos, cfg

model, stoi, itos, cfg = load_or_init_model()

# ----------------------------------------------------------------------
# Text Generation Wrapper
# ----------------------------------------------------------------------
def generate_text(prompt, max_new_tokens, temperature, top_k):
    if not prompt.strip():
        return "Please enter a prompt."
    model.eval()
    encoded = safe_encode(prompt, stoi)
    idx = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        out_idx = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=None if top_k == 0 else int(top_k),
        )
    return safe_decode(out_idx[0].tolist(), itos)

# ----------------------------------------------------------------------
# Generator-based training wrapper for live Gradio progress updates
# ----------------------------------------------------------------------
def train_model(corpus_file, sample_preset, sample_corpus_text, training_steps, batch_size, lr, progress=gr.Progress()):
    global model, stoi, itos, cfg

    # 1. Determine corpus source
    if corpus_file is not None:
        file_path = corpus_file.name if hasattr(corpus_file, "name") else str(corpus_file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif sample_preset == "💬 Multi-turn Dialogue":
        text = DEFAULT_DIALOGUE
    elif sample_preset == "📜 Shakespeare Monologue":
        text = DEFAULT_SHAKESPEARE
    elif sample_corpus_text and sample_corpus_text.strip():
        text = sample_corpus_text
    else:
        yield "❌ Error: Please upload a .txt file or select a sample dataset."
        return

    if len(text.strip()) < 50:
        yield f"❌ Error: Text length ({len(text)} characters) is too short. Please provide at least 100+ characters."
        return

    # Repeat small texts so the model can sample diverse context windows
    while len(text) < 1500:
        text = text + "\n\n" + text

    # 2. Build character vocabulary
    stoi_new, itos_new = get_char_vocab(text)
    unk_idx = stoi_new.get(UNK_CHAR, 0)
    data = torch.tensor([stoi_new.get(c, unk_idx) for c in text], dtype=torch.long)

    # 3. Dynamic block size and batch size scaling
    block_size = min(128, max(16, len(data) - 2))
    batch_size = int(batch_size)
    max_possible_batch = max(2, (len(data) - block_size) // 2)
    if batch_size > max_possible_batch:
        batch_size = max_possible_batch

    total_steps = int(training_steps)

    class CFG: pass
    cfg_new = CFG()
    cfg_new = configure_model(cfg_new)
    cfg_new.vocab_size = len(stoi_new)
    cfg_new.block_size = block_size
    cfg_new.max_steps = total_steps
    cfg_new.learning_rate = float(lr)
    cfg_new.batch_size = batch_size

    yield f"⚙️ Initializing model architecture ({cfg_new.n_layer} layers, {cfg_new.n_head} heads, vocab size {cfg_new.vocab_size}, block size {cfg_new.block_size})..."

    model_local = build_model(cfg_new)
    optimizer = torch.optim.AdamW(
        model_local.parameters(),
        lr=cfg_new.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    def get_batch():
        max_idx = len(data) - block_size
        ix = torch.randint(max_idx, (cfg_new.batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix])
        return x, y

    model_local.train()
    initial_loss = None

    for step in progress.tqdm(range(total_steps), desc="Training Model"):
        xb, yb = get_batch()
        logits = model_local(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1)
        )
        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_local.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % max(1, total_steps // 20) == 0 or step == total_steps - 1:
            status = f"⏳ Step {step+1}/{total_steps} | Loss: {loss.item():.4f} (Initial Loss: {initial_loss:.4f})"
            yield status

    # 4. Save checkpoint
    ckpt = {
        "model_state_dict": model_local.state_dict(),
        "cfg": {k: getattr(cfg_new, k) for k in dir(cfg_new) if not k.startswith("_")},
        "stoi": stoi_new,
        "itos": itos_new,
    }
    torch.save(ckpt, "tinylm.ckpt")

    # 5. Update active global model state in memory
    model = model_local
    model.eval()
    stoi = stoi_new
    itos = itos_new
    cfg = cfg_new

    yield f"✅ Training complete! Checkpoint saved to tinylm.ckpt.\n" \
          f"📉 Loss decreased from {initial_loss:.4f} ➡️ {loss.item():.4f} across {total_steps} steps.\n" \
          f"✨ Active model updated in memory! Switch to the 🧪 Generate tab to test completions."

# ----------------------------------------------------------------------
# Gradio UI Layout
# ----------------------------------------------------------------------
with gr.Blocks(title="Tiny LLM Playground") as demo:
    gr.Markdown("# 🔬 Tiny LLM Playground\n")
    gr.Markdown(
        """
        Explore, train, and test a mini transformer language model (~2.7M parameters).
        
        > **Tip for Character-Level Models:** Character models work best with short, clean prompt tags (`User:` / `Bot:`) and moderate temperature (`0.4` - `0.6`). High temperatures can cause character stuttering (`USISER:` or `you you`).
        """
    )

    with gr.Tab("🧪 Generate"):
        gr.Markdown("### Test Prompt Completion")
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Enter prompt prefix", 
                value="User: Hi\nBot:",
                lines=3
            )
        with gr.Row():
            btn_chat = gr.Button("💬 Chat Prompt ('User: Hi')", size="sm")
            btn_python = gr.Button("🐍 Question ('User: What is Python?')", size="sm")
            btn_shake = gr.Button("📜 Shakespeare ('To be, or not to be')", size="sm")

        btn_chat.click(lambda: "User: Hi\nBot:", None, prompt)
        btn_python.click(lambda: "User: What is Python?\nBot:", None, prompt)
        btn_shake.click(lambda: "To be, or not to be", None, prompt)

        with gr.Row():
            max_tok = gr.Slider(1, 300, value=60, step=1, label="Max new tokens")
            temp = gr.Slider(0.1, 2.0, value=0.5, step=0.05, label="Temperature (0.4-0.6 recommended)")
            topk = gr.Slider(0, 100, value=20, step=1, label="Top-k (0 = off)")
        gen_btn = gr.Button("🚀 Generate Completion", variant="primary")
        output = gr.Textbox(label="Generated Text", lines=8, interactive=False)
        gen_btn.click(
            fn=generate_text,
            inputs=[prompt, max_tok, temp, topk],
            outputs=output,
        )

    with gr.Tab("🚂 Train"):
        gr.Markdown(
            """
            ### Train / Fine-Tune Model
            Select a built-in dataset or upload a `.txt` file. 
            Set **Training Steps to 500+** to achieve low loss and coherent answers.
            """
        )
        with gr.Row():
            sample_preset = gr.Radio(
                ["💬 Multi-turn Dialogue", "📜 Shakespeare Monologue", "✏️ Custom Text below"],
                value="💬 Multi-turn Dialogue",
                label="Preset Dataset"
            )
        with gr.Row():
            corpus_input = gr.File(label="Upload custom corpus (.txt)", file_types=[".txt"])
            sample_input = gr.Textbox(
                label="Custom Corpus Text", 
                value=DEFAULT_DIALOGUE,
                lines=6
            )
        with gr.Row():
            training_steps = gr.Slider(50, 2000, value=500, step=50, label="Training Steps (Iterations)")
            bs = gr.Slider(8, 128, value=32, step=4, label="Batch size")
            lr = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate")
        train_btn = gr.Button("⚡ Start Training", variant="stop")
        train_status = gr.Textbox(label="Training Status & Live Loss", lines=5, interactive=False)
        train_btn.click(
            fn=train_model,
            inputs=[corpus_input, sample_preset, sample_input, training_steps, bs, lr],
            outputs=train_status,
        )

    with gr.Tab("ℹ️ Info"):
        gr.Markdown(
            f"""
            **Current Model Architecture & Config**

            - Layers (`n_layer`): `{getattr(cfg, 'n_layer', '?')}`
            - Heads (`n_head`): `{getattr(cfg, 'n_head', '?')}`
            - Embedding dim (`n_embd`): `{getattr(cfg, 'n_embd', '?')}`
            - Block size (`block_size`): `{getattr(cfg, 'block_size', '?')}`
            - Vocab size (`vocab_size`): `{getattr(cfg, 'vocab_size', '?')}`
            - Dropout: `{getattr(cfg, 'dropout', 0.0)}`

            *Tip:* Edit `Config.py` and restart `app.py` to change transformer architecture layers/heads/dimensions!
            """
        )

# ----------------------------------------------------------------------
# Launch Server
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
