"""TextCNN (Kim, 2014): parallel 1-D convolutions over embeddings + global max-pooling."""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 4, embed_dim: int = 128,
                 kernel_sizes=(2, 3, 4, 5), num_filters: int = 100, dropout: float = 0.5, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):                       # x: [B, L]
        e = self.embedding(x).transpose(1, 2)   # [B, E, L]
        feats = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))
