import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Add this import at the top
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os

DATA_PATH = "data/raw/dataset.csv"
MODEL_PATH = "models/asl_model.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

def train():
    # load dataset
    data = pd.read_csv(DATA_PATH)

    X = data.drop(["label"], axis=1, errors="ignore").values
    y_raw = data["label"].values
    
    # 2. Convert Characters ('A', 'B') to Integers (0, 1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    num_classes = len(np.unique(y))
    
    print(f"Unique classes found: {le.classes_}")
    print(f"Number of classes: {num_classes}")

    # normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler and label encoder for later use in the backend
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # neural network
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
       tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # train
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # evaluate accuracy and loss
    loss, accuracy = model.evaluate(X_test, y_test)

    # confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Neural Network Confusion Matrix")
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to models/confusion_matrix.png")


    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_pred_prob = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(12, 8))

    auc_scores = {}
    for i, class_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = (fpr, tpr, roc_auc)

    # plot only imperfect classes
    for class_name, (fpr, tpr, roc_auc) in auc_scores.items():
        if roc_auc < 1.0:
            ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    # plot macro average
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([
        np.interp(mean_fpr, fpr, tpr)
        for fpr, tpr, _ in auc_scores.values()
    ], axis=0)
    mean_auc = np.mean([v[2] for v in auc_scores.values()])
    ax.plot(mean_fpr, mean_tpr, 'b--', lw=2, label=f"Macro Average (AUC = {mean_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Imperfect Classes Only")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=150)
    plt.show()

    # print all AUC scores
    print("\nAUC Scores per class:")
    for class_name, (_, _, roc_auc) in sorted(auc_scores.items(), key=lambda x: x[1][2]):
        print(f"  {class_name}: {roc_auc:.4f}")

    print(f"\nMean AUC: {mean_auc:.4f}")

    print(f"NN Test Accuracy: {accuracy * 100:.2f}%")
    # print(f"Test loss: {loss:.4f}")


    # save
    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

    print("Test accuracy:", accuracy)
    print("Test loss:", loss)

    # save
    model.save(MODEL_PATH)

    print("Model saved to", MODEL_PATH)

if __name__ == "__main__":
    train()