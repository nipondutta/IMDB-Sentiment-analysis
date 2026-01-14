# IMDB-Sentiment-analysis
IMDB Movie Review Sentiment Analysis

# IMDB Sentiment Analysis using Simple RNN

This project implements **sentiment analysis on the IMDB movie reviews dataset** using a **Simple Recurrent Neural Network (Simple RNN)** built with **TensorFlow / Keras**. The model classifies movie reviews as **positive** or **negative** based on the text content.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that determines the emotional tone behind a body of text. In this project:

* The **IMDB dataset** is used (binary sentiment classification)
* Text reviews are converted into numerical sequences
* An **Embedding layer** learns word representations
* A **Simple RNN** captures sequential dependencies in text
* The model outputs a binary sentiment prediction

---

## 🧠 Model Architecture

The neural network follows this architecture:

1. **Input Layer** – fixed-length padded sequences
2. **Embedding Layer** – converts word indices into dense vectors
3. **Simple RNN Layer** – processes text sequentially
4. **Dense Output Layer** – sigmoid activation for binary classification

```
Input → Embedding → SimpleRNN → Dense (Sigmoid)
```

---

## 🛠️ Technologies Used

* Python 3
* TensorFlow / Keras
* NumPy
* IMDB Dataset (Keras built-in)

---

## 📂 Dataset

* **Dataset**: IMDB Movie Reviews
* **Classes**: Positive (1), Negative (0)
* **Source**: `tensorflow.keras.datasets.imdb`
* **Vocabulary Size**: Configurable (e.g., 10,000 most frequent words)

---

## ⚙️ Preprocessing Steps

* Load IMDB dataset
* Limit vocabulary size
* Pad sequences to a fixed length
* Split into training and validation sets

Padding ensures all input sequences have the same length, which is required for RNN models.

---

## 🚀 Training Details

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy
* **Batch Size**: Typically 16 or 32
* **Epochs**: Up to 30–50 (with Early Stopping)
* **Early Stopping**: Used to prevent overfitting

---

## 📊 Results

* The model learns basic sentiment patterns from text
* Simple RNN provides a baseline for sequence modeling
* Performance can be further improved using LSTM or GRU

---

## ▶️ How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies

   ```bash
   pip install tensorflow numpy
   ```

3. Run the training script or notebook

---

## 🔮 Future Improvements

* Replace Simple RNN with **LSTM / GRU**
* Use **pretrained embeddings** (GloVe, Word2Vec)
* Add **Dropout** for better generalization
* Perform hyperparameter tuning

---

## 📄 License

This project is for **educational purposes**.

---

## 🙌 Acknowledgements

* IMDB Dataset
* TensorFlow / Keras documentation

---

⭐ If you find this project helpful, consider starring the repository!
