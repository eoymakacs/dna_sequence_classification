# 🧬 DNA Sequence Classification – Promoter Detection

## Overview
This project classifies DNA sequences as **promoter** or **non-promoter** using a **Convolutional Neural Network (CNN)**.  
It demonstrates a bioinformatics application of deep learning for sequence-based classification, suitable for detecting gene regulatory regions.

---

## Dataset
The dataset is the **Molecular Biology (Promoter Gene Sequences)** dataset from the **UCI Machine Learning Repository**.

- **Source:** [UCI Promoter Dataset](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Promoter+Gene+Sequences))
- **Samples:** 106 DNA sequences  
- **Sequence Length:** 57 nucleotides each  
- **Columns:**
  - Label (`+` = promoter / `-` = non-promoter)  
  - Sequence name (ignored in training)  
  - DNA sequence  

**Note:** The raw dataset file `promoters.data` should be placed in the `/data` folder. The script automatically converts it to `promoter_sequences.csv` for training.

---

## Project Structure
```plaintext
root/
├── data/
│ └── promoters.data # Raw UCI dataset
├── src/
│ └── model_training.py # Training and evaluation script
└── README.md
```


---

## Preprocessing
- DNA sequences are **one-hot encoded**:
  - `A` → `[1,0,0,0]`  
  - `C` → `[0,1,0,0]`  
  - `G` → `[0,0,1,0]`  
  - `T` → `[0,0,0,1]`  
- Labels are encoded as `positive` (promoter) or `negative` (non-promoter)  
- Sequences with missing data are dropped  

---

## Model Architecture

A simple **1D CNN** is used for sequence classification:

- **Conv1D** layer: 64 filters, kernel size 3, ReLU  
- **MaxPooling1D** layer: pool size 2  
- **Dropout**: 0.2  
- **Conv1D** layer: 128 filters, kernel size 3, ReLU  
- **MaxPooling1D** layer: pool size 2  
- **Flatten**  
- **Dense** layer: 64 units, ReLU  
- **Dense** output layer: 2 units, Softmax  

- **Loss:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  

---

## Training

```bash
cd src
python model_training.py
```

- Batch size: 8
- Epochs: 20
- Validation split: 20%

The script automatically:

**1.** Checks for promoter_sequences.csv
**2.** Converts promoters.data if CSV is missing
**3.** Preprocesses sequences
**4.** Trains the CNN
**5.** Evaluates using accuracy, confusion matrix, and classification report

---

## Results
- Expected Test Accuracy: ~85–95%
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
- Visualization:
  - Confusion Matrix
  - Training / Validation Accuracy over epochs

---

## Insights

- CNNs can detect local motifs in DNA sequences, which is key for identifying promoter regions.
- This pipeline can be extended to:
  - Enhancer vs non-enhancer classification
  - Gene family classification
  - Mutation impact prediction
 
---
 
## 🔮 Next Steps

- Experiment with **RNN architectures** (LSTM or GRU) to capture sequential dependencies in DNA sequences.  
- Explore **k-mer embeddings** instead of one-hot encoding for richer sequence representation.  
- Train and evaluate on **larger genomic datasets** from sources like:
  - [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)  
  - [Kaggle Bioinformatics Datasets](https://www.kaggle.com/datasets?tags=13206-bioinformatics)  
- Apply the pipeline to related bioinformatics problems, such as:
  - Enhancer vs non-enhancer classification  
  - Gene family prediction  
  - Mutation effect prediction  
- Optimize the CNN with **hyperparameter tuning** (filters, kernel sizes, dropout rates) to improve performance.  
- Implement **model explainability techniques** (e.g., saliency maps) to visualize which DNA motifs the model is learning.

---

## 📚 References

- [UCI Machine Learning Repository – Promoter Gene Sequences](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Promoter+Gene+Sequences))  
- [scikit-learn: Label Encoding & Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)  
- [TensorFlow Keras Documentation](https://www.tensorflow.org/guide/keras)  
- [1D Convolutional Neural Networks for Sequence Data](https://keras.io/api/layers/convolution_layers/convolution1d/)  
- [Introduction to DNA Promoter Regions](https://www.ncbi.nlm.nih.gov/books/NBK21899/)


