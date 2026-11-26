"""Grouped DistilBERT and SVM pilot comparisons on development data."""
from pathlib import Path
import time
import os
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from fakejob.data import load_split
from fakejob.evaluation import all_metrics, best_f1_threshold
from fakejob.features import STEP2_CONFIG, build_pipeline, posting_features


def run_pilot(*, use_gpu=False):
    """Train both models once and save their paired validation metrics."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if sklearn.__version__ != "1.6.1":
        raise RuntimeError("Use scikit-learn 1.6.1 for the original grouped split.")
    torch.set_num_threads(1)
    seed = 42
    set_seed(seed)
    with threadpool_limits(limits=1), sklearn.config_context(working_memory=64):
        split = load_split()
    train = split.model_df.loc[split.train_index]
    validation = split.model_df.loc[split.val_index]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_pos, cal_pos = next(folds.split(train, train.fraudulent, split.groups.loc[train.index]))
    fit, calibration = train.iloc[fit_pos], train.iloc[cal_pos]
    # Select one fixed group-stratified subset from each disjoint pool before training.
    def grouped_subset(frame, divisions):
        picker = StratifiedGroupKFold(n_splits=divisions, shuffle=True, random_state=seed)
        _, keep = next(picker.split(frame, frame.fraudulent, split.groups.loc[frame.index]))
        return frame.iloc[keep]

    if use_gpu:
        if not torch.backends.mps.is_available():
            raise RuntimeError("This run needs an Apple GPU with MPS enabled.")
        # Take the first four of the same eight folds; fold zero was the original pilot.
        picker = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
        fit_folds = list(picker.split(fit, fit.fraudulent, split.groups.loc[fit.index]))
        keep = np.sort(np.concatenate([positions for _, positions in fit_folds[:4]]))
        fit = fit.iloc[keep]
    else:
        fit = grouped_subset(fit, 8)
    calibration = grouped_subset(calibration, 4)
    validation = grouped_subset(validation, 4)
    parts = (fit, calibration, validation)
    group_sets = [set(split.groups.loc[part.index]) for part in parts]
    assert all(group_sets[i].isdisjoint(group_sets[j]) for i,j in [(0,1),(0,2),(1,2)])
    assert all(part.index.intersection(split.test_index).empty for part in parts)
    for name, part in zip(["Fit", "Threshold selection", "Evaluation"], parts):
        print(f"{name}: {len(part):,} postings, {part.fraudulent.sum()} fake")

    def evaluate(name, calibration_scores, validation_scores):
        threshold = best_f1_threshold(calibration.fraudulent, calibration_scores)
        metrics = all_metrics(validation.fraudulent, validation_scores, threshold)
        predicted = validation_scores >= threshold
        fake = validation.fraudulent.to_numpy() == 1
        return {"model": name, "threshold": threshold, **metrics,
                "detected": int((fake & predicted).sum()), "missed": int((fake & ~predicted).sum()),
                "false_alarms": int((~fake & predicted).sum()), "evaluation_size": len(validation)}

    with threadpool_limits(limits=1):
        svm = build_pipeline(STEP2_CONFIG).fit(posting_features(fit), fit.fraudulent)
    svm_result = evaluate("Selected SVM", svm.decision_function(posting_features(calibration)),
                          svm.decision_function(posting_features(validation)))

    checkpoint = "distilbert/distilbert-base-uncased"
    revision = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    # Explicit attention supports training dropout on the local Apple GPU.
    attention_options = {"attn_implementation": "eager"} if use_gpu else {}
    bert = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, revision=revision, num_labels=2, **attention_options
    )
    bert.base_model.requires_grad_(False)
    bert.distilbert.transformer.layer[-1].requires_grad_(True)
    device = torch.device("mps" if use_gpu else "cpu")
    bert.to(device)
    print(f"Fine-tuning on {device}", flush=True)

    def encode_postings(frame):
        # Reserve space for every field so a long profile cannot crowd out the description.
        budgets = {"title": 12, "company_profile": 16, "description": 32}
        ids = [[tokenizer.cls_token_id] for _ in range(len(frame))]
        for field, limit in budgets.items():
            tokens = tokenizer(frame[field].fillna("").tolist(), add_special_tokens=False,
                               truncation=True, max_length=limit, return_attention_mask=False)
            for sequence, fragment in zip(ids, tokens["input_ids"]):
                sequence.extend(fragment)
                sequence.append(tokenizer.sep_token_id)
        batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        return TensorDataset(batch["input_ids"], batch["attention_mask"],
                             torch.tensor(frame.fraudulent.to_numpy(), dtype=torch.long))

    fit_data, cal_data, val_data = map(encode_postings, parts)
    counts = np.bincount(fit.fraudulent, minlength=2)
    weights = torch.tensor(len(fit) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW((p for p in bert.parameters() if p.requires_grad), lr=3e-5)
    loader = DataLoader(fit_data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    bert.train()
    for step, (ids, attention, labels) in enumerate(loader, 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logits = bert(input_ids=ids.to(device), attention_mask=attention.to(device)).logits
        loss = loss_fn(logits, labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        optimizer.step()
        if use_gpu:
            # Let completed GPU work settle, then leave an idle interval between batches.
            torch.mps.synchronize()
            time.sleep(0.5 * (time.monotonic() - started))
        if step % 100 == 0 or step == len(loader):
            print(f"Training batch {step}/{len(loader)}", flush=True)

    @torch.inference_mode()
    def distilbert_scores(data):
        bert.eval()
        scores = []
        for ids, attention, _ in DataLoader(data, batch_size=8):
            logits = bert(input_ids=ids.to(device), attention_mask=attention.to(device)).logits
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        return np.concatenate(scores)

    bert_result = evaluate("DistilBERT (last block)", distilbert_scores(cal_data), distilbert_scores(val_data))
    pilot_results = pd.DataFrame([svm_result, bert_result])
    filename = "step4_distilbert_moderate.csv" if use_gpu else "step4_distilbert_pilot.csv"
    pilot_results.to_csv(Path("results") / filename, index=False)
    print("Saved both evaluation rows.")
