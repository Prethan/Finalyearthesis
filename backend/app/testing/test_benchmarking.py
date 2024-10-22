from transformers import BertTokenizer, BertForSequenceClassification
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')


#Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the directory where the fine-tuned BERT model is saved
model_dir = os.path.join(current_directory, '..', 'finetune_models', 'ft_bert_sentiment_analysis')

# Load fine-tuned BERT model and tokenizer
tokenizer_finetuned  = BertTokenizer.from_pretrained(model_dir)
model_finetuned = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3)  # 3 labels: positive, negative, neutral

# # Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer_bert = BertTokenizer.from_pretrained(model_name)
model_bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: positive, negative, neutral

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment using NLTK VADER
def analyze_sentiment_vader(text):
    # Get sentiment scores
    scores = sid.polarity_scores(text)
    
    # Determine sentiment label based on compound score
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment

# Function to evaluate dataset using NLTK VADER sentiment analysis
def evaluate_dataset_vader(dialogues, sentiments):
    predicted_sentiments_vader = []

    for dialogue in dialogues:
        sentiment = analyze_sentiment_vader(dialogue)
        predicted_sentiments_vader.append(sentiment)

    accuracy_vader = accuracy_score(sentiments, predicted_sentiments_vader)
    precision_vader = precision_score(sentiments, predicted_sentiments_vader, average='weighted')
    recall_vader = recall_score(sentiments, predicted_sentiments_vader, average='weighted')
    f1_vader = f1_score(sentiments, predicted_sentiments_vader, average='weighted')
    cm_vader = confusion_matrix(sentiments, predicted_sentiments_vader)

    return accuracy_vader, precision_vader, recall_vader, f1_vader, cm_vader, predicted_sentiments_vader

# Function to tokenize and preprocess text data
def preprocess_text(text, tokenizer):
    return tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, truncation=True,
                                  padding='max_length', return_tensors='pt')

def evaluate_dataset(model, tokenizer, dialogues, sentiments):
    # Initialize lists to store predictions and probabilities
    predicted_sentiments = []
    predicted_probabilities = []

    # Iterate through each dialogue in the dataset
    for dialogue in dialogues:
        # Tokenize the dialogue
        inputs = tokenizer(dialogue, padding=True, truncation=True, max_length=512, return_tensors="pt")

        # Predict sentiment using the model
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Map predicted class to sentiment label
        label_map_inverse = {0: "Negative", 1: "Neutral", 2: "Positive"}
        predicted_sentiment = label_map_inverse[predicted_class]

        # Append predicted sentiment and probabilities
        predicted_sentiments.append(predicted_sentiment)
        predicted_probabilities.append(probabilities.tolist())

    # Calculate evaluation metrics
    accuracy = accuracy_score(sentiments, predicted_sentiments)
    precision = precision_score(sentiments, predicted_sentiments, average='weighted')
    recall = recall_score(sentiments, predicted_sentiments, average='weighted')
    f1 = f1_score(sentiments, predicted_sentiments, average='weighted')
    cm = confusion_matrix(sentiments, predicted_sentiments)

    return accuracy, precision, recall, f1, cm


# Load dataset from CSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    dialogues = df['dialogue'].tolist()
    sentiments = df['sentiment'].tolist()
    return dialogues, sentiments

# Load dataset from CSV file
def load_dataset_movie(file_path):
    df = pd.read_csv(file_path)
    dialogues = df['text'].tolist()
    sentiments = df['label'].tolist()
    return dialogues, sentiments

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the original dataset CSV file 
csv_file_path_dialogues = os.path.join(current_directory, '..', 'datasets', 'dialogues_sentiment.csv')
dialogues, sentiments = load_dataset(csv_file_path_dialogues)

csv_file_path_movies = os.path.join(current_directory, '..', 'datasets', 'movie.csv')
movies, labels = load_dataset_movie(csv_file_path_movies)

# Evaluate dataset using the fine-tuned BERT model
accuracy_finetuned, precision_finetuned, recall_finetuned, f1_finetuned, cm_finetuned = evaluate_dataset(model_finetuned, tokenizer_finetuned, dialogues, sentiments)

# Evaluate dataset using the BERT model
accuracy_bert, precision_bert, recall_bert, f1_bert, cm_bert = evaluate_dataset(model_bert, tokenizer_bert, dialogues, sentiments)

# Evaluate dataset using NLTK VADER sentiment analysis
accuracy_nltk, precision_nltk, recall_nltk, f1_nltk, cm_nltk, predicted_sentiments_vader = evaluate_dataset_vader(dialogues, sentiments)

print("\n********************* FINED TUNED BERT SENTIMENT ANALYSIS MODEL **************************\n")

# Display evaluation metrics
print("EVALUATION METRICS:")

# Format data
data = [
    ["Accuracy", f"{accuracy_finetuned * 100:.2f}%"],
    ["Precision", f"{precision_finetuned:.3f}"],
    ["Recall", f"{recall_finetuned:.3f}"],
    ["F1 Score", f"{f1_finetuned:.3f}"]
]

# Print table
print(tabulate(data, headers=["Evaluation Metrics", "Value"], tablefmt="grid"))

# Display Confusion Matrix
print("\nCONFUSION MATRIX:")
for row in cm_finetuned:
    print("[", "  ".join(str(cell) for cell in row), "]")


print("\n\n")


print("\n********************* BERT SENTIMENT ANALYSIS MODEL **************************\n")

# Display evaluation metrics
print("EVALUATION METRICS:")

# Format data
data = [
    ["Accuracy", f"{accuracy_bert * 100:.2f}%"],
    ["Precision", f"{precision_bert:.3f}"],
    ["Recall", f"{recall_bert:.3f}"],
    ["F1 Score", f"{f1_bert:.3f}"]
]

# Print table
print(tabulate(data, headers=["Evaluation Metrics", "Value"], tablefmt="grid"))

# Display Confusion Matrix
print("\nCONFUSION MATRIX:")
for row in cm_bert:
    print("[", "  ".join(str(cell) for cell in row), "]")


print("\n\n")

print("\n********************* FINED TUNED BERT SENTIMENT ANALYSIS MODEL **************************\n")

# Display evaluation metrics
print("EVALUATION METRICS:")

# Format data
data = [
    ["Accuracy", f"{accuracy_nltk * 100:.2f}%"],
    ["Precision", f"{precision_nltk:.3f}"],
    ["Recall", f"{recall_nltk:.3f}"],
    ["F1 Score", f"{f1_nltk:.3f}"]
]

# Print table
print(tabulate(data, headers=["Evaluation Metrics", "Value"], tablefmt="grid"))

# Display Confusion Matrix
print("\nCONFUSION MATRIX:")
for row in cm_nltk:
    print("[", "  ".join(str(cell) for cell in row), "]")


print("\n\n")