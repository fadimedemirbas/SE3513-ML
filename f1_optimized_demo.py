
import numpy as np
import pandas as pd
import joblib
import sklearn

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from fakejob.data import load_split
from fakejob.features import STEP2_CONFIG, posting_features
from fakejob.evaluation import model_scores, best_f1_threshold


# ==========================================
# 1. SETTINGS
# ==========================================

RANDOM_STATE = 42
N_SPLITS = 5

MODEL_PATH = "fakejob_f1_optimized_demo.joblib"


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading dataset...")

split = load_split()

X_all = posting_features(split.model_df)

X = X_all.loc[split.dev_index]
y = split.y.loc[split.dev_index]
groups = split.groups.loc[split.dev_index]

print("Dataset loaded.")
print("Development samples:", len(X))
print("Fake postings:", int(y.sum()))
print("Real postings:", int((y == 0).sum()))


# ==========================================
# 3. CREATE MODEL
# ==========================================

print("\nCreating optimized SVM model...")

config = STEP2_CONFIG.but(C=3.0)

model = __import__(
    "fakejob.features",
    fromlist=["build_pipeline"]
).build_pipeline(config)


# ==========================================
# 4. GROUPED CROSS VALIDATION
# ==========================================

print("\nStarting Grouped Cross Validation...")

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_scores = np.empty(len(X), dtype=float)

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):

    print(f"\nFold {fold}/{N_SPLITS}")

    fold_model = clone(model)

    fold_model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    scores = model_scores(
        fold_model,
        X.iloc[val_idx]
    )

    oof_scores[val_idx] = scores

    print("Fold completed.")


# ==========================================
# 5. THRESHOLD OPTIMIZATION
# ==========================================

print("\nOptimizing decision threshold...")

threshold = best_f1_threshold(
    y.to_numpy(),
    oof_scores
)

print("Best threshold:", threshold)


# ==========================================
# 6. COMPARE DEFAULT AND OPTIMIZED THRESHOLD
# ==========================================

default_predictions = (oof_scores >= 0).astype(int)

optimized_predictions = (
    oof_scores >= threshold
).astype(int)


def show_metrics(name, y_true, predictions):

    print(f"\n{name}")

    print(
        "F1 Score:",
        round(f1_score(y_true, predictions, zero_division=0), 4)
    )

    print(
        "Precision:",
        round(precision_score(y_true, predictions, zero_division=0), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_true, predictions, zero_division=0), 4)
    )


show_metrics(
    "DEFAULT THRESHOLD (0)",
    y,
    default_predictions
)

show_metrics(
    "OPTIMIZED THRESHOLD",
    y,
    optimized_predictions
)


# ==========================================
# 7. TRAIN FINAL DEMO MODEL
# ==========================================

print("\nTraining final model...")

final_model = clone(model)

final_model.fit(X, y)


# ==========================================
# 8. SAVE MODEL
# ==========================================

model_data = {

    "pipeline": final_model,

    "threshold": float(threshold),

    "config": {
        "algorithm": "LinearSVC",
        "C": 3.0,
        "word_ngrams": (1, 2),
        "char_ngrams": (3, 5),
        "class_weight": "balanced"
    },

    "sklearn_version": sklearn.__version__

}

joblib.dump(model_data, MODEL_PATH)

print("\nModel saved successfully!")
print("File:", MODEL_PATH)


# ==========================================
# 9. DEMO PREDICTION FUNCTION
# ==========================================

def predict_job(posting):

    defaults = {
        "title": "",
        "company_profile": "",
        "description": "",
        "requirements": "",
        "benefits": "",
        "employment_type": "Unknown",
        "required_experience": "Unknown",
        "required_education": "Unknown",
        "function": "Unknown",
        "industry": "Unknown",
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "salary_range": np.nan,
        "location": "Unknown"
    }

    defaults.update(posting)

    data = pd.DataFrame([defaults])

    features = posting_features(data)

    score = model_scores(
        model_data["pipeline"],
        features
    )[0]

    prediction = int(score >= model_data["threshold"])

    print("\n========== JOB ANALYSIS ==========")

    print("Decision score:", round(float(score), 4))
    print("Decision threshold:", round(threshold, 4))

    if prediction == 1:
        print("RESULT: FAKE JOB POSTING")
    else:
        print("RESULT: LIKELY REAL JOB POSTING")

    print("=================================")

    return prediction


# ==========================================
# 10. DEMO EXAMPLES
# ==========================================

real_job = {

    "title": "Junior Software Developer",

    "company_profile": (
        "An established software development company."
    ),

    "description": (
        "We are looking for a junior software developer. "
        "The candidate will work with our development team "
        "to develop and maintain software applications."
    ),

    "requirements": (
        "Knowledge of Python, Java and SQL. "
        "Good communication and teamwork skills."
    ),

    "employment_type": "Full-time",

    "required_experience": "Entry level",

    "has_company_logo": 1

}


fake_job = {

    "title": "Work From Home - Earn Money Immediately",

    "company_profile": "",

    "description": (
        "Earn thousands of dollars every day without "
        "experience. Guaranteed income. Pay a registration "
        "fee immediately to start working."
    ),

    "requirements": "",

    "employment_type": "Contract",

    "required_experience": "Not Applicable",

    "telecommuting": 1,

    "has_company_logo": 0

}


print("\n\nDEMO TEST 1")

predict_job(real_job)

print("\n\nDEMO TEST 2")

predict_job(fake_job)


print("\nDemo completed successfully!")
