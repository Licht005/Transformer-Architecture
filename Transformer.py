# Full Transformer Implementation in PyTorch


import torch
import torch.nn as nn


# 1. Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure embedding dimension divides evenly into heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by number of heads"

        # Linear layers for Queries, Keys, Values
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final linear transformation
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]   # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape into (batch, seq_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Compute energy scores: QK^T
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention weights
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)

        # Weighted sum of values
        out = torch.einsum("nhqk,nkhd->nqhd", attention, values).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)



# 2. Transformer Block

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Position-wise Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Apply attention + residual connection + normalization
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        # Apply feed-forward + residual connection + normalization
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


# 3. Encoder

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device,
                 forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Embeddings for words + positions
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Input embedding = word embedding + positional embedding
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Pass through each Transformer block
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


# 4. Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # Masked self-attention
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))

        # Cross-attention with encoder output
        out = self.transformer_block(value, key, query, src_mask)
        return out


# 5. Decoder
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads,
                 forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack of decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        # Final linear layer to get predictions
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Input embeddings
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Pass through decoder blocks
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


# 6. Full Transformer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 embed_size=256, num_layers=6, forward_expansion=4, heads=8,
                 dropout=0, device="cuda", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads,
                               device, forward_expansion, dropout, max_length)

        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads,
                               forward_expansion, dropout, device, max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Mask padding tokens in source
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # Prevent decoder from attending to future tokens
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # Encode source sequence
        enc_src = self.encoder(src, src_mask)

        # Decode using encoder output + target
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# -------------------------------
# Quick Test Run
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example vocab sizes (e.g., 10000 words)
    src_vocab_size = 10000
    trg_vocab_size = 10000
    src_pad_idx = 0
    trg_pad_idx = 0

    # Create Transformer
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
        embed_size=256, num_layers=6, forward_expansion=4,
        heads=8, dropout=0.1, device=device, max_length=100
    ).to(device)

    # Dummy input (batch_size=2, sequence_length=10)
    src = torch.randint(0, src_vocab_size, (2, 10)).to(device)
    trg = torch.randint(0, trg_vocab_size, (2, 10)).to(device)

    # Forward pass
    out = model(src, trg)
    print("Output shape:", out.shape)  # Expected: (batch_size, trg_len, trg_vocab_size)
