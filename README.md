# IMDB Sentiment Analysis using DistilBERT and Hugging Face Transformers

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.x-yellow.svg)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) <!-- Choose a license -->

## Overview

This project demonstrates how to perform sentiment analysis on the IMDB movie review dataset using a pre-trained DistilBERT model from the Hugging Face Hub. The goal is to classify movie reviews as either positive or negative. This is a common benchmark task in Natural Language Processing (NLP) and serves as a practical example of fine-tuning Transformer models for text classification.

The project leverages the `transformers` and `datasets` libraries from Hugging Face for efficient data loading, tokenization, model fine-tuning, and evaluation. It follows standard practices learned from courses like Stanford's CS224N.

## Features

*   Loads the standard IMDB dataset using Hugging Face `datasets`.
*   Preprocesses and tokenizes text data specifically for DistilBERT.
*   Fine-tunes the `distilbert-base-uncased` model for binary sequence classification.
*   Utilizes the Hugging Face `Trainer` API for streamlined training and evaluation.
*   Evaluates the model performance using Accuracy and F1-score.
*   Provides scripts/notebook for training, evaluation, and prediction on new text.

## Technologies Used

*   **Python:** 3.8+
*   **PyTorch:** Deep learning framework
*   **Hugging Face Libraries:**
    *   `transformers`: For pre-trained models (DistilBERT), tokenizers, and the `Trainer` API.
    *   `datasets`: For loading and preprocessing the IMDB dataset.
*   **Scikit-learn:** For calculating evaluation metrics (Accuracy, F1-score).
*   **NumPy / Pandas:** For numerical operations and data handling (optional exploration).
*   **(Optional) Google Colab:** Used for development and training with free GPU access.

## Results

The fine-tuned DistilBERT model achieved the following performance on the IMDB test set (25,000 reviews) after 3 epochs of training:

*   **Accuracy:** ~92.94%
*   **F1 Score (Binary):** ~92.86%
*   **Evaluation Loss:** ~0.208

These results indicate a strong ability of the model to accurately classify the sentiment of unseen movie reviews.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Link to your GitHub repository]
    cd [Your repository folder name]
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    **(Make sure you have created a `requirements.txt` file - see note below)**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hardware:** A GPU is highly recommended for faster training, although fine-tuning DistilBERT might be feasible (but slow) on a CPU. This code was tested on Google Colab with a GPU runtime.

## Usage

**(Adapt these instructions based on how you structure your code: e.g., separate Python scripts or a single Jupyter/Colab Notebook)**

**1. Training:**

*   **(If using scripts):** Run the training script. This will typically load data, preprocess it, fine-tune the model, and save the best checkpoint.
    ```bash
    python train.py [--output_dir ./sentiment_model_results] [--epochs 3] [--batch_size 16]
    ```
*   **(If using a Notebook):** Execute the cells in the notebook sequentially, starting from data loading, preprocessing, defining `TrainingArguments`, initializing the `Trainer`, and calling `trainer.train()`.

**2. Evaluation:**

*   Evaluation is performed automatically during training (if `evaluation_strategy` is set) and at the end if using the `Trainer`.
*   To evaluate a saved model on the test set:
    **(If using scripts):**
    ```bash
    python evaluate.py --model_path ./final_sentiment_model --test_data [path_to_test_data_or_load_logic]
    ```
    **(If using Notebook):** Run the cells that load the test dataset, preprocess it, and call `trainer.evaluate(tokenized_test_dataset)`.

**3. Prediction:**

*   Use the fine-tuned model to predict the sentiment of new movie reviews:
    **(If using scripts):**
    ```bash
    python predict.py --model_path ./final_sentiment_model --text "This movie was surprisingly good!"
    ```
    **(If using Notebook):** Use the `predict_sentiment` function defined in the notebook, ensuring the model and tokenizer are loaded first.
    ```python
    # Example within the notebook:
    # from your_module import predict_sentiment # Or define function in notebook
    # result = predict_sentiment("This movie was surprisingly good!")
    # print(result) # Output: {'label': 'Positive', 'confidence': 0.9XXX}
    ```
