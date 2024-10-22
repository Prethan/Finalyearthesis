from transformers import BertTokenizer, BertForSequenceClassification
import os
import torch
from sklearn.metrics import accuracy_score

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the directory where the fine-tuned BERT model is saved
model_dir = os.path.join(current_directory, '..', 'finetune_models', 'ft_bert_sentiment_analysis')
print(model_dir)

# Load fine-tuned BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=3)  # 3 labels: positive, negative, neutral

# Function to tokenize and preprocess text data
def preprocess_text(text):
    return tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, truncation=True,
                                  padding='max_length', return_tensors='pt')

# Define a function to test individual accuracy
def test_individual_accuracy(dialogue, sentiment):
    # Tokenize the dialogue
    encoded_dialogue = preprocess_text(dialogue)
    
    # Convert sentiment to label_map
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    sentiment_label = label_map[sentiment]
    
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate the dialogue
    model.eval()
    with torch.no_grad():
        input_ids = encoded_dialogue['input_ids'].to(device)
        attention_mask = encoded_dialogue['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    predicted_sentiment = predicted.item()
    label_map_inverse = {v: k for k, v in label_map.items()}
    predicted_sentiment_str = label_map_inverse[predicted_sentiment]
    
    # Calculate accuracy as a percentage
    accuracy_percentage = 100 * (predicted_sentiment == sentiment_label)
    
    # Calculate confidence score
    confidence_score = probabilities[0][predicted_sentiment].item()
    
    return predicted_sentiment_str, accuracy_percentage, confidence_score

# Test individual accuracy for a dialogue
dialogue_1 = "I think this just might be my masterpiece."
sentiment_1 = "Positive"

dialogue_2 = "I'm so sick of this. I hate this feeling, like I'm being torn from the inside out."
sentiment_2 = "Negative"

dialogue_3 = "Make your passion your profession, and work will become play."
sentiment_3 = "Positive"

dialogue_4 = "I'm not sure what to think about that."
sentiment_4 = "Neutral"

dialogue_5 = "I can't believe how terrible the service was at that restaurant."
sentiment_5 = "Negative"

predicted_sentiment, accuracy_percentage, confidence_score = test_individual_accuracy(dialogue_1, sentiment_1)
print("Dialogue 1")
print("Predicted Sentiment:", predicted_sentiment)
print("Accuracy (%):", accuracy_percentage)
print("Confidence Score:", confidence_score)

predicted_sentiment, accuracy_percentage, confidence_score = test_individual_accuracy(dialogue_2, sentiment_2)
print("Dialogue 2")
print("Predicted Sentiment:", predicted_sentiment)
print("Accuracy (%):", accuracy_percentage)
print("Confidence Score:", confidence_score)

predicted_sentiment, accuracy_percentage, confidence_score = test_individual_accuracy(dialogue_3, sentiment_3)
print("Dialogue 3")
print("Predicted Sentiment:", predicted_sentiment)
print("Accuracy (%):", accuracy_percentage)
print("Confidence Score:", confidence_score)

predicted_sentiment, accuracy_percentage, confidence_score = test_individual_accuracy(dialogue_4, sentiment_4)
print("Dialogue 4")
print("Predicted Sentiment:", predicted_sentiment)
print("Accuracy (%):", accuracy_percentage)
print("Confidence Score:", confidence_score)

predicted_sentiment, accuracy_percentage, confidence_score = test_individual_accuracy(dialogue_5, sentiment_5)
print("Dialogue 5")
print("Predicted Sentiment:", predicted_sentiment)
print("Accuracy (%):", accuracy_percentage)
print("Confidence Score:", confidence_score)
