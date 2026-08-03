import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score


df = pd.read_csv("fusion_final_features.csv")

X = df.drop(columns=["Participant_ID", "PHQ8_Binary"])
y = df["PHQ8_Binary"]


X = X.iloc[:, :100]

print("Using features:", X.shape[1])
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        C=0.5,
        penalty='l2',
        max_iter=1000
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='accuracy'
)

f1_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring=make_scorer(f1_score)
)

print("\n===== FUSION MODEL RESULTS =====")
print("Accuracy Mean:", np.mean(accuracy_scores))
print("Accuracy Std :", np.std(accuracy_scores))
print("F1 Score Mean:", np.mean(f1_scores))
print("F1 Score Std :", np.std(f1_scores))

# Save results
np.savetxt("results/accuracy_scores.txt", accuracy_scores)
np.savetxt("results/f1_scores.txt", f1_scores)

print("\nExecution completed successfully.")
