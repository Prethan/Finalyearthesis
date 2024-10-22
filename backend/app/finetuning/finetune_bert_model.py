from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: positive, negative, neutral

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dialogues, sentiments, label_map, tokenizer, max_length):
        self.dialogues = dialogues
        self.sentiments = sentiments
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = str(self.dialogues[idx])
        sentiment = self.label_map[self.sentiments[idx]]
        encoding = self.tokenizer(dialogue, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {'dialogue': dialogue, 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'sentiment': torch.tensor(sentiment)}

    def collate_fn(self, batch):
        dialogues = [item['dialogue'] for item in batch]
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        sentiments = torch.stack([item['sentiment'] for item in batch])
        return {'dialogue': dialogues, 'input_ids': input_ids, 'attention_mask': attention_mask, 'sentiment': sentiments}

# Function to tokenize and preprocess text data
def preprocess_text(text):
    return tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, truncation=True,
                                  padding='max_length', return_tensors='pt')

# Function to fine-tune BERT model on custom dataset
def fine_tune_bert(dialogues, sentiments, epochs=10, batch_size=8, learning_rate=2e-5):
    # Define label mapping
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}

    # Split dataset into training and validation sets
    train_dialogues, val_dialogues, train_sentiments, val_sentiments = train_test_split(
        dialogues, sentiments, test_size=0.1, random_state=42)
    
    max_length = 512  # Define the maximum sequence length

    # Create DataLoader for training and validation sets
    train_dataset = CustomDataset(train_dialogues, train_sentiments, label_map, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = CustomDataset(val_dialogues, val_sentiments, label_map, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the selected device
    model.to(device)

    # Fine-tune BERT model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['sentiment'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_predictions = []
        val_probabilities = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['sentiment'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_probabilities.extend(torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_accuracy = correct / total
            avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # Convert numeric predictions to string labels
        label_map_inverse = {v: k for k, v in label_map.items()}
        val_predictions_str = [label_map_inverse[label] for label in val_predictions]

        # Calculate evaluation metrics
        evaluate_model(val_sentiments, val_predictions_str, val_probabilities)

    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Define the directory where you want to save the fine-tuned model
    output_dir = os.path.join(current_directory, '..', 'finetune_models', 'ft_bert_sentiment_analysis')

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Function to evaluate the model
def evaluate_model(true_labels, predicted_labels, predicted_probabilities):
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print("Recall:", recall)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print("F1 Score:", f1)


# Load dataset from CSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    dialogues = df['dialogue'].tolist()
    sentiments = df['sentiment'].tolist()
    return dialogues, sentiments

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the original dataset CSV file 
csv_file_path = os.path.join(current_directory, '..', 'datasets', 'dialogues_sentiment.csv')
dialogues, sentiments = load_dataset(csv_file_path)
fine_tune_bert(dialogues, sentiments)
