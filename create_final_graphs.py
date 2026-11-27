
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Load final test predictions
df = pd.read_csv("results/final_test_predictions.csv")

y_true = df["y_true"]
y_pred = df["y_pred"]
y_prob = df["fake_probability_platt"]

# Load final metrics
with open("results/final_test_metrics.json", "r") as file:
    report = json.load(file)

metrics = report["metrics"]

# Create output directory
import os
os.makedirs("results/figures", exist_ok=True)

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Fake"]
)
display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Final Test Confusion Matrix")
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix.png", dpi=300)
plt.close()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = roc_auc_score(y_true, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Final Test")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("results/figures/roc_curve.png", dpi=300)
plt.close()

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall, precision, label=f"Average Precision = {ap:.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - Final Test")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("results/figures/precision_recall_curve.png", dpi=300)
plt.close()

# 4. Bootstrap Confidence Intervals
ci = pd.read_csv("results/final_test_group_bootstrap_ci.csv")

fig, ax = plt.subplots(figsize=(9, 6))

positions = range(len(ci))
estimates = ci["estimate"]
lower_errors = estimates - ci["ci_low"]
upper_errors = ci["ci_high"] - estimates

ax.errorbar(
    estimates,
    positions,
    xerr=[lower_errors, upper_errors],
    fmt="o",
    capsize=5
)

ax.set_yticks(list(positions))
ax.set_yticklabels(ci["metric"])
ax.set_xlabel("Metric value")
ax.set_title("95% Group Bootstrap Confidence Intervals")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig("results/figures/bootstrap_confidence_intervals.png", dpi=300)
plt.close()

print("All final evaluation graphs created successfully!")

print("\nFinal metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nSaved figures:")
print("results/figures/confusion_matrix.png")
print("results/figures/roc_curve.png")
print("results/figures/precision_recall_curve.png")
print("results/figures/bootstrap_confidence_intervals.png")
