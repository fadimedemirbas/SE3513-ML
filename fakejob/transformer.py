"""Fine-tuning DistilBERT on job postings for Step 5 (distilbert_ensemble.ipynb).

The settings are fixed in advance (see the notebook) and are not tuned. Every call of
`fine_tune_and_score()` starts from a fresh pretrained model, trains it on the given
rows only and returns scores for other rows, so it can be used inside grouped
cross-validation exactly like a scikit-learn estimator.
"""
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup, set_seed

CHECKPOINT = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"

# Tokens reserved per field so that a long company profile cannot crowd out the description.
# 1 [CLS] + 24 + 64 + 164 + 3 [SEP] = 256 tokens.
FIELD_BUDGETS = {"title": 24, "company_profile": 64, "description": 164}
MAX_LENGTH = 1 + sum(FIELD_BUDGETS.values()) + len(FIELD_BUDGETS)


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    eval_batch_size: int = 64
    max_grad_norm: float = 1.0


def encode_postings(tokenizer, frame):
    """`[CLS] title [SEP] company profile [SEP] description [SEP]`, each field truncated to its budget.

    Returns `input_ids` and `attention_mask` tensors of shape (n_postings, MAX_LENGTH).
    """
    ids = [[tokenizer.cls_token_id] for _ in range(len(frame))]
    for field, limit in FIELD_BUDGETS.items():
        tokens = tokenizer(frame[field].fillna("").tolist(), add_special_tokens=False,
                           truncation=True, max_length=limit, return_attention_mask=False)
        for sequence, fragment in zip(ids, tokens["input_ids"]):
            sequence.extend(fragment)
            sequence.append(tokenizer.sep_token_id)
    batch = tokenizer.pad({"input_ids": ids}, padding="max_length", max_length=MAX_LENGTH,
                          return_tensors="pt")
    return batch["input_ids"], batch["attention_mask"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def _trim(ids, mask):
    """Drop padding columns that no posting in the batch uses (faster, same result)."""
    length = int(mask.sum(dim=1).max())
    return ids[:, :length], mask[:, :length]


def fine_tune_and_score(make_model, input_ids, attention_mask, y, fit_rows, score_rows,
                        settings: TrainSettings, seed: int, device) -> np.ndarray:
    """Fine-tune a fresh model on `fit_rows` and return fraud scores for `score_rows`.

    The score is logit(fake) - logit(real), so higher means more likely fake, like the SVM's
    decision function. The loss is weighted by inverse class frequency in `fit_rows`.
    """
    _seed_everything(seed)
    y = np.asarray(y)
    model = make_model().to(device)
    use_amp = settings.fp16 and device.type == "cuda"

    labels = torch.tensor(y[fit_rows], dtype=torch.long)
    counts = np.bincount(y[fit_rows], minlength=2)
    weights = torch.tensor(len(fit_rows) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(
        TensorDataset(input_ids[fit_rows], attention_mask[fit_rows], labels),
        batch_size=settings.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameters = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": settings.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate)
    total_steps = settings.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(settings.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for _ in range(settings.epochs):
        for ids, mask, target in loader:
            ids, mask = _trim(ids, mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), target.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(score_rows), settings.eval_batch_size):
            rows = score_rows[start:start + settings.eval_batch_size]
            ids, mask = _trim(input_ids[rows], attention_mask[rows])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(scores).astype(float)
