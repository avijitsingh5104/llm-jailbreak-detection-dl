"""Bidirectional LSTM with masked attention pooling over the hidden states."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 4, embed_dim: int = 128, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.4, pooling: str = "attn", pad_id: int = 0):
        super().__init__()
        assert pooling in ("attn", "max", "last")
        self.pad_id, self.pooling = pad_id, pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, return_attn: bool = False):            # x: [B, L]
        mask = x != self.pad_id                                  # [B, L]
        lengths = mask.sum(1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))   # [B, L, 2H]
        weights = None
        if self.pooling == "attn":
            scores = self.attn(out).squeeze(-1).masked_fill(~mask, float("-inf"))
            scores = torch.where(mask.any(1, keepdim=True), scores, torch.zeros_like(scores))
            weights = torch.softmax(scores, dim=1)                                  # [B, L]
            pooled = (out * weights.unsqueeze(-1)).sum(1)
        elif self.pooling == "max":
            pooled = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
            pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        else:  # last: final forward + backward states of the top layer
            pooled = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(self.dropout(pooled))
        return (logits, weights) if return_attn else logits
