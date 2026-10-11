import random
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

CHARS = "HPIT"
CONTROL_TOKENS = ("<PAD>", "<BOS>", "<SEP>", "<EOS>")

#return balanced input
def balance(source: str) -> str:
    if any(c not in CHARS for c in source):
        raise ValueError(f"Input must contain only characters from {CHARS}")
    if not source:
        return ""
    counts = Counter(source)
    order = list(dict.fromkeys(source))#quick way to get unique characters in order of first appearance
    maximum = max(counts.values())
    return "".join(c * maximum for c in order)


def worst_target_length(length: int) -> int:
    if length < 0:
        raise ValueError("length cannot be negative")
    if length == 0:
        return 0
    return max(k * (length - k + 1) for k in range(1, min(len(CHARS), length) + 1))


def max_full_sequence_length(max_input_length: int) -> int:
    if max_input_length < 0:
        raise ValueError("max_input_length cannot be negative")
    return max_input_length + 3 + worst_target_length(max_input_length)


class Tokenizer:
    tokens = CONTROL_TOKENS + tuple(CHARS)

    def __init__(self) -> None:
        self.to_id = {token: i    for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.sep_id, self.eos_id = range(len(CONTROL_TOKENS))

    def encode_chars(self, source: str) -> list[int]:
        if any(c not in CHARS for c in source):
            raise ValueError(f"Only characters from {CHARS} are valid data characters")
        return [self.to_id[c] for c in source]

    def decode_chars(self, ids: Iterable[int]) -> str:
        return "".join(self.tokens[i] for i in ids if i >= len(CONTROL_TOKENS))

    def prompt(self, source: str) -> list[int]:
        balance(source)
        return [self.bos_id, *self.encode_chars(source), self.sep_id]

    def example(self, source: str) -> tuple[list[int], list[int]]:
        prompt = self.prompt(source)
        full = [*prompt, *self.encode_chars(balance(source)), self.eos_id]
        inputs, labels = full[:-1], full[1:]
        labels[: len(prompt) - 1] = [-100] * (len(prompt) - 1)
        return inputs, labels


class BalanceDataset(Dataset):
    def __init__(self, sources: list[str], tokenizer: Tokenizer):
        self.sources = sources
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int):
        x, y = self.tokenizer.example(self.sources[index])
        return torch.tensor(x), torch.tensor(y)


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(x) for x, _ in batch)
    x = torch.full((len(batch), maximum), pad_id, dtype=torch.long)
    y = torch.full((len(batch), maximum), -100, dtype=torch.long)
    for row, (ids, labels) in enumerate(batch):
        x[row, : len(ids)] = ids
        y[row, : len(labels)] = labels
    return x, y #x and y are both padded to the row * maximum


def _candidate(rng: random.Random, length: int, extreme: bool = False) -> str:
    if length == 0:
        return ""
    k = rng.randint(1, min(len(CHARS), length))#choose how many different characters to use
    chars = rng.sample(CHARS, k)
    if k == 1:
        return chars[0] * length
    style = "extreme" if extreme else rng.choice(("balanced", "mild", "extreme"))
    counts = [1] * k
    remaining = length - k
    if style == "extreme":
        counts[0] += remaining
    elif style == "balanced":
        for i in range(remaining):
            counts[i % k] += 1
    else:
        weights = [rng.uniform(0.5, 2.0) for _ in range(k)]
        for chosen in rng.choices(range(k), weights=weights, k=remaining):
            counts[chosen] += 1
    symbols = [c for c, count in zip(chars, counts) for _ in range(count)]
    rng.shuffle(symbols)
    return "".join(symbols)

#only way you define seed
def generate_unique(
    count: int, min_length: int, max_length: int, seed: int,
    excluded: set[str] | None = None, extreme: bool = False,
) -> list[str]:
    if count < 1 or min_length < 0 or max_length < min_length:
        raise ValueError("Invalid sample count or length range")
    rng = random.Random(seed)
    seen = set() if excluded is None else set(excluded)#to mainly quicken duplicate checking other than that, it is not necessary
    result: list[str] = []
    attempts = 0
    
    while len(result) < count and attempts < max(10000, count * 300):
        length = min_length + (attempts % (max_length - min_length + 1))
        candidate = _candidate(rng, length, extreme)
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
        attempts += 1
    if len(result) != count:
        raise ValueError(f"Only {len(result)} unique inputs available; requested {count}")
    rng.shuffle(result)
    return result


@dataclass
class Splits:
    train: list[str]
    val: list[str]
    iid: list[str]
    ood_17_24: list[str]
    ood_25_32: list[str]
    stress: list[str]
def make_splits(train_size=6000, val_size=1000, test_size=1000, stress_size=200,
                seed=42, train_max_length=16) -> Splits:
    if not 16 <= train_max_length <= 32:
        raise ValueError("train_max_length must be between 16 and 32")
    #reserve some short sequences to ensure that each split has at least one example of each length up to 4
    reserve_count = min(len(CHARS), train_size - 1, val_size, test_size)
    reserved = [[], [], []]
    rng = random.Random(seed)
    for length in range(1, reserve_count + 1):
        candidates = ["".join(parts) for parts in itertools.product(CHARS, repeat=length)]
        rng.shuffle(candidates)
        for group in range(3):
            reserved[group].append(candidates[group])
    blocked = set(reserved[0] + reserved[1] + reserved[2])

    groups = []
    for index, size in enumerate((train_size, val_size, test_size)):
        group = reserved[index].copy()
        if index == 0:
            group.insert(0, "")#the sole empty input belongs to training.
        if size > len(group):
            group += generate_unique(size - len(group), 1, 16, seed + index, blocked)
        blocked.update(group)
        groups.append(group)
    train, val, iid = groups

    ood_a = generate_unique(test_size, 17, 24, seed + 3)
    ood_b = generate_unique(test_size, 25, 32, seed + 4)
    stress = generate_unique(stress_size, 17, 32, seed + 5, set(ood_a + ood_b), extreme=True)
    if train_max_length != 16:
        # Freeze the original validation/test sets, including newly in-range lengths.
        # Exclude them before drawing the extended training set to prevent leakage.
        held_out = set(val + iid + ood_a + ood_b + stress)
        prefix = train[:reserve_count + 1]  # Empty input and reserved short examples.
        train = prefix + generate_unique(train_size - len(prefix), 1, train_max_length,
                                         seed, held_out | set(prefix))
        if set(train) & held_out:
            raise RuntimeError("Extended training set overlaps a held-out split")
    return Splits(train, val, iid, ood_a, ood_b, stress)
