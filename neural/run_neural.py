import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
from pathlib import Path

from neural.datasets import StanceDataset
from neural.models import BertBaseline, BertLateFusion
from neural.train import train_one_epoch, evaluate
from collections import defaultdict
from sklearn.metrics import f1_score

@torch.no_grad()
def evaluate_with_topic(model, loader, device):
    model.eval()
    gold, pred, topics = [], [], []
    for batch in loader:
        # topic 不用 to(device)
        t = batch.get("topic", None)
        if t is None:
            raise ValueError("Dataset must return 'topic' in batch")
        topics.extend(list(t))

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items() if k != "topic"}

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

    # overall
    overall_acc = sum(int(a==b) for a,b in zip(gold, pred)) / max(1, len(gold))
    overall_f1  = f1_score(gold, pred, average="macro")

    # per-topic
    idx_by_topic = defaultdict(list)
    for i, tp in enumerate(topics):
        idx_by_topic[tp].append(i)

    per_topic = {}
    for tp, idxs in idx_by_topic.items():
        g = [gold[i] for i in idxs]
        p = [pred[i] for i in idxs]
        acc = sum(int(a==b) for a,b in zip(g,p)) / max(1, len(g))
        f1  = f1_score(g, p, average="macro")
        per_topic[tp] = {"n": len(idxs), "acc": acc, "macro_f1": f1}

    return overall_acc, overall_f1, per_topic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--backbone", default="bert-base-uncased")
    ap.add_argument("--model_type", choices=["N1", "N2", "N3", "N2RS"], default="N1")
    ap.add_argument("--rf_path", default=None)
    ap.add_argument("--sf_path", default=None)
    ap.add_argument("--rf_dim", type=int, default=0)
    ap.add_argument("--sf_dim", type=int, default=0)
    ap.add_argument("--num_labels", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--test", default=None, help="path to test.jsonl (optional)")
    ap.add_argument("--save_dir", default="../checkpoints", help="where to save best model")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = StanceDataset(args.train, args.backbone, args.max_len,
                             rhetoric_feat_path=args.rf_path if args.model_type in ["N2","N2RS"] else None,
                             sentiment_feat_path=args.sf_path if args.model_type in ["N3"] else None)
    dev_ds = StanceDataset(args.dev, args.backbone, args.max_len,
                           rhetoric_feat_path=args.rf_path if args.model_type in ["N2","N2RS"] else None,
                           sentiment_feat_path=args.sf_path if args.model_type in ["N3"] else None)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.bs)

    if args.model_type == "N1":
        model = BertBaseline(args.backbone, args.num_labels)
    elif args.model_type == "N2":
        model = BertLateFusion(args.backbone, args.num_labels, rf_dim=args.rf_dim, sf_dim=0)
    elif args.model_type == "N3":
        model = BertLateFusion(args.backbone, args.num_labels, rf_dim=0, sf_dim=args.sf_dim)
    else:
        # 这里留给你实现 N2RS：BERT logits + rhetoric logits 融合（最快）
        raise NotImplementedError("N2RS(logits fusion) 你若要我也可以给完整实现。")

    model.to(device)

    optim = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = os.path.join(
        args.save_dir,
        f"best_{args.model_type}_{args.backbone.replace('/', '-')}.pt"
    )

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, sched, device)
        dev_acc, dev_f1 = evaluate(model, dev_loader, device)

        improved = dev_f1 > best_f1
        if improved:
            best_f1 = dev_f1
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "best_dev_f1": best_f1,
                    "args": vars(args),
                },
                best_path
            )

        flag = " ✅ best" if improved else ""
        print(f"epoch={ep} loss={loss:.4f} dev_acc={dev_acc:.4f} dev_f1={dev_f1:.4f}{flag}")
    print(f"\nBest dev_f1={best_f1:.4f} saved to: {best_path}")
# ========= 2. 训练结束后：加载 best → 跑 test =========
    if args.test:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(
            f"\nLoaded best checkpoint "
            f"(epoch={ckpt['epoch']}, dev_f1={ckpt['best_dev_f1']:.4f})"
        )

        test_ds = StanceDataset(
            args.test,
            args.backbone,
            args.max_len,
            rhetoric_feat_path=args.rf_path if args.model_type in ["N2"] else None,
            sentiment_feat_path=args.sf_path if args.model_type in ["N3"] else None,
        )
        test_loader = DataLoader(test_ds, batch_size=args.bs)

        #test_acc, test_f1 = evaluate(model, test_loader, device)
        #print(f"TEST  acc={test_acc:.4f}  macro_f1={test_f1:.4f}")

        test_acc, test_f1, per_topic = evaluate_with_topic(model, test_loader, device)
        print(f"TEST  acc={test_acc:.4f}  macro_f1={test_f1:.4f}")

        # 打印 per-topic（按样本数排序）
        for tp, m in sorted(per_topic.items(), key=lambda x: -x[1]["n"]):
            print(f"  [{tp}] n={m['n']} acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
