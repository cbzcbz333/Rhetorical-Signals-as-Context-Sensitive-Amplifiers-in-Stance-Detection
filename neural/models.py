import torch
import torch.nn as nn
from transformers import AutoModel

class BertBaseline(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]  # [CLS]
        logits = self.cls(self.drop(h))
        return logits

class BertLateFusion(nn.Module):
    """
    N2: [CLS] + rhetoric (and optional sentiment) → classifier
    """
    def __init__(self, backbone: str, num_labels: int, rf_dim: int, sf_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden + rf_dim + sf_dim, num_labels)

    def forward(self, input_ids, attention_mask, rf=None, sf=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]
        feats = [h]
        if rf is not None:
            feats.append(rf)
        if sf is not None:
            feats.append(sf)
        x = torch.cat(feats, dim=-1)
        logits = self.cls(self.drop(x))
        return logits
