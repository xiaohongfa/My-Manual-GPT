import argparse
import json
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import BalanceDataset, Tokenizer, balance, collate, make_splits, max_full_sequence_length, worst_target_length
from model import GPT, GPTConfig


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return torch.device("cuda")
        print(f"CUDA device sm_{major}{minor} is unsupported by this PyTorch build; using CPU.", flush=True)
    return torch.device("cpu")


def loader(sources: list[str], tokenizer: Tokenizer, batch_size: int,
           shuffle: bool = False) -> DataLoader:
    return DataLoader(
        BalanceDataset(sources, tokenizer), batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate, pad_id=tokenizer.pad_id),
    )


def teacher_forced(model: GPT, data_loader: DataLoader,
                   device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = correct = tokens = 0
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            valid = y.ne(-100)
            count = int(valid.sum())
            total_loss += float(loss) * count
            correct += int(((logits.argmax(-1) == y) & valid).sum())
            tokens += count
    return {"loss": total_loss / tokens, "token_accuracy": correct / tokens}


@torch.inference_mode()
def generated_sources(model: GPT, sources: list[str], tokenizer: Tokenizer,
                      device: torch.device, batch_size: int = 1):
    if batch_size < 1:
        raise ValueError("generation batch size must be positive")
    if batch_size == 1:
        for source in sources:
            yield source, model.generate(tokenizer.prompt(source), tokenizer.eos_id,
                                         worst_target_length(len(source)) + 1, device)
        return
    # Equal input lengths give equal prompt lengths: no padding or position changes.
    # Finished rows are ignored; causal attention never mixes different batch rows.
    model.eval()
    buckets = defaultdict(list)
    for source in sources:
        buckets[len(source)].append(source)
    for length, bucket in sorted(buckets.items()):
        for offset in range(0, len(bucket), batch_size):
            selected = bucket[offset:offset + batch_size]
            ids = torch.tensor([tokenizer.prompt(s) for s in selected], dtype=torch.long, device=device)
            finished = [False] * len(selected)
            outputs = [[] for _ in selected]
            for _ in range(worst_target_length(length) + 1):
                logits, _ = model(ids)
                next_ids = logits[:, -1].argmax(-1)
                for row, value in enumerate(next_ids.tolist()):
                    if not finished[row]:
                        outputs[row].append(value)
                        finished[row] = value == tokenizer.eos_id
                if all(finished):
                    break
                ids = torch.cat((ids, next_ids[:, None]), dim=1)
            yield from zip(selected, outputs)


def free_running(model: GPT, sources: list[str], tokenizer: Tokenizer,
                 device: torch.device, generation_batch_size: int = 1) -> dict[str, Any]:
    exact = length_ok = eos_ok = 0
    by_length = defaultdict(lambda: [0, 0])
    for source, output in generated_sources(model, sources, tokenizer, device, generation_batch_size):
        target = tokenizer.encode_chars(balance(source))
        exact += output == [*target, tokenizer.eos_id]
        eos_ok += len(output) == len(target) + 1 and output[-1] == tokenizer.eos_id
        emitted = output[:-1] if output and output[-1] == tokenizer.eos_id else output
        length_ok += len(emitted) == len(target)
        by_length[len(source)][0] += output == [*target, tokenizer.eos_id]
        by_length[len(source)][1] += 1
    n = len(sources)
    return {
        "exact_match_accuracy": exact / n,
        "output_length_accuracy": length_ok / n,
        "eos_accuracy": eos_ok / n,
        "by_length": {str(k): {"exact_match_accuracy": v[0] / v[1], "count": v[1]}
                      for k, v in sorted(by_length.items())},
    }


def save_plot(results: dict[str, dict[str, Any]], path: Path, train_max_length: int = 16) -> None:
    plt.figure(figsize=(9, 4.5))
    for name in ("iid", "ood_17_24", "ood_25_32", "stress"):
        points = results[name]["by_length"]
        x = sorted(map(int, points))
        y = [points[str(length)]["exact_match_accuracy"] for length in x]
        plt.plot(x, y, marker="o", label=name if train_max_length == 16 else name.replace("ood_", "test_"))
    plt.axvline(train_max_length + 0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Input length")
    plt.ylabel("Exact Match Accuracy")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = device_for_training()
    tokenizer = Tokenizer()
    data_seed = args.seed if args.data_seed is None else args.data_seed
    splits = make_splits(args.train_size, args.val_size, args.test_size, args.stress_size,
                         seed=data_seed, train_max_length=args.train_max_length)
    config = GPTConfig(
        n_layers=args.n_layers,
        max_seq_len=max_full_sequence_length(32),#where you input the max length of the input sequence
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_loader = loader(splits.train, tokenizer, args.batch_size, shuffle=True)
    val_loader = loader(splits.val, tokenizer, args.batch_size)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    training_started = time.perf_counter()
    best_val = float("inf")
    print(f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, context={config.max_seq_len}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = token_count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(y.ne(-100).sum())
            loss_sum += float(loss.detach()) * count
            token_count += count
        val = teacher_forced(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / token_count, "val_loss": val["loss"],
               "val_token_accuracy": val["token_accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"config": model.config_dict(), "state_dict": model.state_dict(),
                        "seed": args.seed, "data_seed": data_seed, "train_max_length": args.train_max_length,
                        "sizes": [args.train_size, args.val_size, args.test_size, args.stress_size]},
                       out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "timing.json").write_text(json.dumps({"training_seconds": time.perf_counter() - training_started}), encoding="utf-8")
    evaluate_checkpoint(args)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    out = Path(args.output)
    path = out / "best.pt" if args.command == "train" else Path(args.checkpoint)
    device = device_for_training()
    model, checkpoint = load_checkpoint(path, device)
    tokenizer = Tokenizer()
    splits = make_splits(*checkpoint["sizes"], seed=checkpoint.get("data_seed", checkpoint["seed"]),
                         train_max_length=checkpoint.get("train_max_length", 16))
    groups = {"iid": splits.iid, "ood_17_24": splits.ood_17_24,
              "ood_25_32": splits.ood_25_32, "stress": splits.stress}
    results = {}
    parity_checked = set()
    for name, sources in groups.items():
        metrics = teacher_forced(model, loader(sources, tokenizer, args.batch_size), device)
        metrics.update(free_running(model, sources, tokenizer, device, args.generation_batch_size))
        if args.generation_batch_size > 1:
            # Sample every evaluated input length and compare raw tokens to scalar generation.
            probe_buckets = defaultdict(list)
            for source in sources:
                length = len(source)
                if length not in parity_checked and len(probe_buckets[length]) < args.generation_batch_size:
                    probe_buckets[length].append(source)
            probes = [source for bucket in probe_buckets.values() for source in bucket]
            for source, output in generated_sources(model, probes, tokenizer, device, args.generation_batch_size):
                if len(source) in parity_checked:
                    continue
                scalar = model.generate(tokenizer.prompt(source), tokenizer.eos_id,
                                        worst_target_length(len(source)) + 1, device)
                if output != scalar:
                    raise RuntimeError("Batched generation differs from scalar on a parity probe")
                parity_checked.add(len(source))
        results[name] = metrics
        print(name, {k: v for k, v in metrics.items() if k != "by_length"}, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plot(results, out / "exact_match_by_length.png", checkpoint.get("train_max_length", 16))
    (out / "evaluation_verification.json").write_text(json.dumps({
        "generation_batch_size": args.generation_batch_size,
        "scalar_parity_lengths": sorted(parity_checked),
        "scalar_parity_passed": True if parity_checked else None,
    }, indent=2), encoding="utf-8")


def predict(args: argparse.Namespace) -> None:
    tokenizer = Tokenizer()
    device = device_for_training()
    model, _ = load_checkpoint(Path(args.checkpoint), device)
    ids = model.generate(tokenizer.prompt(args.source), tokenizer.eos_id,
                         worst_target_length(len(args.source)) + 1, device)
    emitted = ids[:-1] if ids and ids[-1] == tokenizer.eos_id else ids
    print("predicted:", tokenizer.decode_chars(emitted))
    print("expected: ", balance(args.source))
    print("EOS:", bool(ids and ids[-1] == tokenizer.eos_id))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser("train")
    training.add_argument("--output", default="runs/baseline")
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--train-size", type=int, default=6000)
    training.add_argument("--val-size", type=int, default=1000)
    training.add_argument("--test-size", type=int, default=1000)
    training.add_argument("--stress-size", type=int, default=200)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--n-layers", type=int, default=2)
    training.add_argument("--data-seed", type=int, default=None)
    training.add_argument("--train-max-length", type=int, choices=range(16, 33), default=16)
    training.add_argument("--generation-batch-size", type=int, default=1)

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", default="runs/baseline/best.pt")
    evaluation.add_argument("--output", default="runs/baseline")
    evaluation.add_argument("--batch-size", type=int, default=128)
    evaluation.add_argument("--generation-batch-size", type=int, default=1)

    prediction = sub.add_parser("predict")
    prediction.add_argument("source")
    prediction.add_argument("--checkpoint", default="runs/baseline/best.pt")
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
