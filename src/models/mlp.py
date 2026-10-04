"""MLP baseline on TF-IDF word + character n-gram features (no sequence modelling)."""
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 4, hidden=(512, 128), dropout: float = 0.4):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
