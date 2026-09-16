import numpy as np
import pandas as pd
from features import FEATURE_NAMES
from tqdm import tqdm

np.random.seed(42)

NUM_CLASSES = 36
SAMPLES_PER_CLASS = 100
FRAMES_PER_SAMPLE = 20

def generate_data():
    data = []
    labels = []
    for sign in tqdm(range(NUM_CLASSES), desc="Generating classes"):
        base = np.random.uniform(0.2, 0.8, len(FEATURE_NAMES))
        for i in tqdm(range(SAMPLES_PER_CLASS), desc=f"Class {sign}", leave=False):
            frames = []
            for j in range(FRAMES_PER_SAMPLE):
                noise = np.random.normal(0, 0.05, len(FEATURE_NAMES))
                frame = base + noise
                frames.append(frame)
            frames = np.array(frames)
            flattened = frames.flatten()
            data.append(flattened)
            labels.append(sign)

    columns = []
    for f in range(FRAMES_PER_SAMPLE):
        for feature in FEATURE_NAMES:
            columns.append(f"{feature}_t{f}")

    df = pd.DataFrame(data, columns=columns)
    df["label"] = labels
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print("Dataset generated.")