"""Fine-tuned pretrained Transformer encoder (DistilBERT / RoBERTa-base) for prompt classification."""
import torch
import torch.nn as nn


class HFClassifier(nn.Module):
    """Thin wrapper so the shared trainer can call model(batch_dict) -> logits."""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 4):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, batch):
        return self.model(**batch).logits


class TokenizeCollator:
    """Tokenises a list of (text, label) pairs with dynamic padding."""

    def __init__(self, tokenizer, max_len: int = 256):
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        texts, labels = zip(*batch)
        enc = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return dict(enc), torch.tensor(labels, dtype=torch.long)
