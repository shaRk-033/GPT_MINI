import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, vocab_size, context_length, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.inp_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(dropout)
        posits = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        v_emb = torch.arange(0, self.d_model, 2, dtype=torch.float)

        posits[:, 0::2] = torch.sin(position / (10000 ** (v_emb / self.d_model)))
        posits[:, 1::2] = torch.cos(position / (10000 ** (v_emb / self.d_model)))

        posits = posits.unsqueeze(0)
        self.register_buffer('posits', posits)

    def forward(self, x):
        x = self.inp_embedding(x)
        x = x + self.posits[:, :x.size(1), :]
        x = self.dropout(x)
        return x

class ResidualLink(nn.Module):
    def __init__(self, sublayer, d_model, dropout):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.sublayer(self.norm(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, head_size, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(self.d_model, self.head_size)
        self.key = nn.Linear(self.d_model, self.head_size)
        self.value = nn.Linear(self.d_model, self.head_size)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out1 = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        out1 = out1.masked_fill(mask == 0, float('-inf'))
        out1 = F.softmax(out1, dim=-1)
        out1 = self.dropout(out1)
        out2 = out1 @ v
        return out2

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_size = self.d_model // self.num_heads
        self.attention_heads = nn.ModuleList([CausalSelfAttention(self.d_model, self.head_size, dropout) for _ in range(self.num_heads)])
        self.w_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        self.out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        self.out = self.w_out(self.out)
        self.out = self.dropout(self.out)
        return self.out

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_model * 4
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out = self.dropout(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention = ResidualLink(MultiHeadAttention(self.d_model, self.num_heads, dropout), self.d_model, dropout)
        self.feedforward = ResidualLink(FeedForwardNetwork(self.d_model, dropout), self.d_model, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, vocab_size, d_model, context_length, n_decoder_blocks, dropout) -> None:
        super().__init__()
        self.embeds = EmbeddingLayer(d_model, vocab_size, context_length, dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, 8, dropout) for _ in range(n_decoder_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embeds(x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

class BigramLM(nn.Module):
    def __init__(self, context_length, n_decoder_blocks, dropout, vocab_size=1000, d_model=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.model = GPTBlock(vocab_size, d_model, context_length, n_decoder_blocks, dropout)
        self.context_length = context_length

    def forward(self, x, targets=None):
        x = x.cuda()
        logits = self.model(x)
        loss = None
        if targets is not None:
            targets = targets.cuda()
            logits = logits.view(-1, self.vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, x, num_char=1000):
        for i in range(num_char):
            x_cond = x[:, -self.context_length:]
            pred, _ = self(x_cond)
            preds = pred[:, -1, :]
            probs = F.softmax(preds, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx), dim=1)
        return x
