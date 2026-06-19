
 #DecodeLabs - Industrial Training Kit (Batch 2026)
 #Project 2: Data Classification Using AI



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)


# STEP 1: LOAD AND UNDERSTAND THE DATASET  ("Raw Material")

iris = load_iris()

X = iris.data                
y = iris.target                
feature_names = iris.feature_names
target_names = iris.target_names

# Put it in a DataFrame just to LOOK at the data clearly
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print(f"Samples : {df.shape[0]}")
print(f"Features: {df.shape[1] - 1}")
print(f"Classes : {df['species'].nunique()} -> {list(target_names)}")
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df["species"].value_counts())


# STEP 2: FEATURE SCALING  ("The Gatekeeper Rule")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# STEP 3: TRAIN-TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y,
)

print("\n" + "=" * 60)
print("STEP 2 & 3: SCALING + TRAIN-TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# STEP 4: CHOOSE THE BEST K  ("Tuning the Engine")

error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    pred_temp = knn_temp.predict(X_test)
    error_rates.append(np.mean(pred_temp != y_test))

best_k = k_range[np.argmin(error_rates)]

print("\n" + "=" * 60)
print("STEP 4: CHOOSING OPTIMAL K")
print("=" * 60)
print(f"Best K found: {best_k} (lowest error rate = {min(error_rates):.4f})")

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker="o", linestyle="--", color="darkorange")
plt.axvline(best_k, color="navy", linestyle=":", label=f"Best K = {best_k}")
plt.title("Choosing K: Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.savefig("k_elbow_curve.png", dpi=150)
plt.close()

# STEP 5: TRAIN THE MODEL  ("The Workflow: Scikit-Learn")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 60)
print("STEP 5 & 6: MODEL TRAINING + EVALUATION")
print("=" * 60)
print(f"Model: KNeighborsClassifier(n_neighbors={best_k})")
print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nFull Classification Report:\n")
print(classification_report(y_test, predictions, target_names=target_names))

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (K={best_k}, Accuracy={acc:.2%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# STEP 7: TEST WITH A NEW / UNSEEN SAMPLE

# Example: a new flower measurement
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # looks like a Setosa
new_sample_scaled = scaler.transform(new_sample)
new_prediction = model.predict(new_sample_scaled)

print("\n" + "=" * 60)
print("STEP 7: PREDICTION ON NEW DATA")
print("=" * 60)
print(f"New sample (raw): {new_sample[0]}")
print(f"Predicted species: {target_names[new_prediction[0]]}")

print("\nDone. Plots saved as 'k_elbow_curve.png' and 'confusion_matrix.png'.")
