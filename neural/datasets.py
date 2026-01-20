import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_feature_map(path: str, key="rf") -> Dict[str, List[float]]:
    m = {}
    for row in load_jsonl(path):
        m[row["id"]] = row[key]
    return m

class StanceDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_len: int = 128,
        rhetoric_feat_path: Optional[str] = None,
        rhetoric_key: str = "rf",
        sentiment_feat_path: Optional[str] = None,
        sentiment_key: str = "sf",
    ):
        self.data = load_jsonl(data_path)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_len = max_len

        self.rf_map = load_feature_map(rhetoric_feat_path, rhetoric_key) if rhetoric_feat_path else None
        self.sf_map = load_feature_map(sentiment_feat_path, sentiment_key) if sentiment_feat_path else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        text = item["text"]
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
            "id": item["id"],
        }

       # if self.rf_map is not None:
            #out["rf"] = torch.tensor(self.rf_map[item["id"]], dtype=torch.float)

        if self.sf_map is not None:
            out["sf"] = torch.tensor(self.sf_map[item["id"]], dtype=torch.float)

        if self.rf_map is not None:
            rid = item["id"]
            if rid not in self.rf_map:
                raise KeyError(f"Missing rf for id={rid}")
            out["rf"] = torch.tensor(self.rf_map[rid], dtype=torch.float)

        out["topic"] = item.get("topic", "UNK")

        return out
