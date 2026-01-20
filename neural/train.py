import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, optim, sched, device, log_every=50):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            rf=batch.get("rf", None),
            sf=batch.get("sf", None),
        )
        loss = F.cross_entropy(logits, batch["labels"])
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if sched:
            sched.step()
        total_loss += loss.item()
        if step % log_every == 0:
            print(f"  step={step}/{len(loader)} loss={loss.item():.4f}", flush=True)
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    gold, pred = [], []
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            rf=batch.get("rf", None),
            sf=batch.get("sf", None),
        )
        p = logits.argmax(-1).cpu().tolist()
        y = batch["labels"].cpu().tolist()
        pred.extend(p)
        gold.extend(y)

    acc = sum(int(a == b) for a, b in zip(gold, pred)) / max(1, len(gold))
    macro_f1 = f1_score(gold, pred, average="macro")
    return acc, macro_f1
