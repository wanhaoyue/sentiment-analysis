import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Load model + tokenizer from Hugging Face Hub
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Label mapping (from HuggingFace cardiffnlp model)
labels = ['Negative', 'Neutral', 'Positive']

st.title("Twitter Sentiment Analysis (Roberta-base)")

user_input = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    # Preprocess text (tokenization)
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    
    # Run prediction
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].softmax(dim=0).cpu().numpy()
    
    # Display results
    for i, (label, score) in enumerate(zip(labels, scores)):
        st.write(f"{label}: {score:.4f}")
    
    # Best prediction
    best_idx = np.argmax(scores)
    st.subheader(f"Predicted Sentiment: {labels[best_idx]}")
