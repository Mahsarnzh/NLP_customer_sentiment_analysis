import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained BERT model for sentiment analysis
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to perform sentiment analysis using BERT
def analyze_sentiment_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    predicted_sentiment = sentiment_labels[predicted_class]
    return predicted_sentiment, probabilities

# Example customer feedback
customer_feedback = "I absolutely love the new features! The customer service is amazing."

# Perform sentiment analysis
predicted_sentiment, sentiment_probabilities = analyze_sentiment_bert(customer_feedback)

# Display results
print("Customer Feedback:", customer_feedback)
print("Predicted Sentiment:", predicted_sentiment)
print("Sentiment Probabilities:", sentiment_probabilities.tolist())
