import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
import matplotlib.pyplot as plt


DATA_PATH = "data/raw/dataset.csv"
MODEL_PATH  = "models/asl_model.joblib"
SCALER_PATH = "models/scaler.joblib"
LABEL_PATH  = "models/label_encoder.joblib"

FEATURES = [
    "thumb_bent", "index_bent", "middle_bent", "ring_bent", "pinky_bent",
    "accel_x", "accel_y", "accel_z",
    "heading", "rolling", "pitch",
    "thumb_touch", "index_touch", "middle_touch"
]

def train():
    data = pd.read_csv(DATA_PATH)
    # data = data[~data["label"].str.match(r'^\d+$')]

    X = data[FEATURES].values
    print(np.shape(X))
    y_raw = data["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_knn.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to models/confusion_matrix_knn.png")

    print(f"KNN Test Accuracy: {acc * 100:.2f}%")

    scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"Average Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")

    unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [le.classes_[i] for i in unique_test_classes]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_test_classes))

    # ROC curves — KNN uses predict_proba
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_pred_prob = knn.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(12, 8))
    auc_scores = {}
    for i, class_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = (fpr, tpr, roc_auc)

    for class_name, (fpr, tpr, roc_auc) in auc_scores.items():
        if roc_auc < 1.0:
            ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in auc_scores.values()], axis=0)
    mean_auc = np.mean([v[2] for v in auc_scores.values()])
    ax.plot(mean_fpr, mean_tpr, 'b--', lw=2, label=f"Macro Average (AUC = {mean_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Imperfect Classes Only")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("models/roc_curve_knn.png", dpi=150)
    plt.show()
    print(f"Mean AUC : {mean_auc:.4f}")
    print(f"KNN Test Accuracy: {acc * 100:.2f}%")
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    # print(f"Average Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")
    

    # save
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_PATH)
    print(f"\nModel and preprocessing tools saved to models/ directory.")

if __name__ == "__main__":
    train()