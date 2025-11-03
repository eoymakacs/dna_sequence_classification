"""
Goal:
Train a model to classify DNA sequences into categories — for example, distinguishing promoter vs non-promoter sequences (or other biological functions).

Dataset:
We'll use the Promoter Gene Sequences dataset from the UCI Machine Learning Repository:

Dataset: Molecular Biology (Promoter Gene Sequences)
    - 106 DNA sequences
    - Each sequence has 57 nucleotides
    - Classes: Positive (promoter) or Negative (non-promoter)
"""

"""
model_training.py
-----------------
DNA Promoter Sequence Classification using CNN.
This script:
1. Checks for promoter_sequences.csv
2. Downloads and converts the UCI dataset if not found
3. Preprocesses sequences (one-hot encoding)
4. Trains a CNN model
5. Evaluates performance and visualizes results
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Assuming model_training.py is in /src and data is in /data at root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT_DIR, "data", "promoters.data")
CSV_FILE = os.path.join(ROOT_DIR, "data", "promoter_sequences.csv")

print(f"ROOT_DIR: {ROOT_DIR}")
print(f"DATA_FILE: {DATA_FILE}")

# -------------------------------------------------------------
# STEP 1: Prepare Dataset
# -------------------------------------------------------------

def prepare_dataset():
    """Convert promoters.data (3-column CSV) to promoter_sequences.csv for training."""
    print("🔍 Checking for dataset...")

    if os.path.exists(CSV_FILE):
        print(f"✅ Found {CSV_FILE}. Loading...")
        df = pd.read_csv(CSV_FILE)
        return df

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please place promoters.data in /data folder.")

    print(f"⚙️ Converting {DATA_FILE} to CSV...")

    # Read CSV-like file with three columns
    df_raw = pd.read_csv(DATA_FILE, header=None, names=['label', 'name', 'sequence'])
    
    # Strip spaces
    df_raw['label'] = df_raw['label'].str.strip()
    df_raw['sequence'] = df_raw['sequence'].str.strip()

    # Convert + / - to positive / negative
    df_raw['label'] = df_raw['label'].map({'+': 'positive', '-': 'negative'})

    # Keep only sequences with non-empty values
    df = df_raw[['sequence', 'label']].dropna()
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Created {CSV_FILE} with {len(df)} sequences.")

    return df

# -------------------------------------------------------------
# STEP 2: Preprocess Data
# -------------------------------------------------------------
def one_hot_encode(seq):
    mapping = {'A': [1,0,0,0],
               'C': [0,1,0,0],
               'G': [0,0,1,0],
               'T': [0,0,0,1]}
    return np.array([mapping.get(nuc, [0,0,0,0]) for nuc in seq])

def preprocess_data(df):
    df['label'] = df['label'].str.strip()
    df['sequence'] = df['sequence'].str.strip()
    
    if df.empty:
        raise ValueError("DataFrame is empty. Check the CSV file.")

    X = np.array([one_hot_encode(seq) for seq in df['sequence']])
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    if len(y) == 0:
        raise ValueError("No labels found. Check 'label' column in CSV.")
    
    y_cat = to_categorical(y)
    print(f"✅ Encoded {X.shape[0]} sequences. Shape: {X.shape}")
    return X, y_cat, le

# -------------------------------------------------------------
# STEP 3: Build CNN Model
# -------------------------------------------------------------
def build_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------------------------------------
# STEP 4: Train and Evaluate
# -------------------------------------------------------------
def train_and_evaluate(model, X_train, y_train, X_test, y_test, label_encoder):
    history = model.fit(X_train, y_train, epochs=20, batch_size=8,
                        validation_split=0.2, verbose=1)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n🎯 Test Accuracy: {acc:.2f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n Classification Report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Accuracy Curve
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# -------------------------------------------------------------
# STEP 5: Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Starting DNA Sequence Classification Pipeline...\n")

    df = prepare_dataset()
    X, y, le = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(input_shape=(X.shape[1], 4))
    train_and_evaluate(model, X_train, y_train, X_test, y_test, le)

    print("\n Training complete. Model ready.")
