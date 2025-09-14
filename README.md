# Transformer (PyTorch Implementation)

## 📌 Overview
This repository contains a **from-scratch implementation of the Transformer architecture** in PyTorch, inspired by the original paper *"Attention Is All You Need"* (Vaswani et al., 2017).  

The code implements the **core building blocks** of the Transformer model, including:
- Multi-Head Self Attention  
- Transformer Encoder & Decoder Blocks  
- Positional Embeddings  
- Masking (padding mask & causal mask)  
- Full Encoder–Decoder Transformer model  

It serves as both an **educational reference** and a **foundation** for experimenting with sequence-to-sequence tasks such as machine translation, text summarization, and more.

---

## ⚡ Features
- Built with **PyTorch**  
- Modular design (easy to extend or modify)  
- Clear comments for learning purposes  
- End-to-end **forward pass** implemented  
- Output verified with random input sequences  

---

## 📂 Files
- `Transformer.py` → Main implementation of the Transformer model.  

---

## 🛠 Usage

### 1. Import the model
```python
import torch
from Transformer import Transformer

# Hyperparameters
src_vocab_size = 10000
trg_vocab_size = 10000
src_pad_idx = 0
trg_pad_idx = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_size=256,
    num_layers=6,
    heads=8,
    forward_expansion=4,
    dropout=0.1,
    device=device,
    max_length=100
).to(device)
```

### 2. Forward pass with dummy data
```python
src = torch.randint(0, src_vocab_size, (2, 10)).to(device)  # batch of 2, seq length 10
trg = torch.randint(0, trg_vocab_size, (2, 10)).to(device)

out = model(src, trg[:, :-1])  # teacher forcing (predict next token)
print("Output shape:", out.shape)  # (batch_size, seq_len, vocab_size)
```

Expected output:
```
Output shape: torch.Size([2, 9, 10000])
```

---

## 📖 Next Steps
This implementation provides the **architecture only**.  
To train on a real dataset, you need:
- A tokenized dataset (e.g., translation pairs).  
- A training loop with `nn.CrossEntropyLoss`.  
- A decoding method (greedy search, beam search, etc.).  

---

## 🎯 Applications
- Machine Translation  
- Text Summarization  
- Question Answering  
- Dialogue Systems / Chatbots  
- As a foundation to build GPT-style models  

---

## 📜 References
- Vaswani et al. (2017). *Attention Is All You Need*. [Paper](https://arxiv.org/abs/1706.03762)  
- PyTorch Documentation: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)  
- Alladin Persson: Pytorch Transformers from Scratch (Attention is all you need)
