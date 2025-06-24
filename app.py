import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()
labels = ['Negative', 'Neutral', 'Positive']

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬")
st.title("💬 Twitter Sentiment Analysis (Roberta-base)")
st.write("Analyze the sentiment of a tweet using a state-of-the-art Transformer model. Try it below:")

# Example tweets
examples = [
    "I love the new movie, it was fantastic!",
    "Service was terrible, very disappointed.",
    "It is going to rain tomorrow.",
]

example_choice = st.selectbox("Choose example tweet:", ["(Type your own)"] + examples)
if example_choice != "(Type your own)":
    user_input = example_choice
else:
    user_input = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].softmax(dim=0).cpu().numpy()

    best_idx = np.argmax(scores)
    sentiment = labels[best_idx]
    color = {"Positive": "green", "Neutral": "yellow", "Negative": "red"}

    st.markdown(f"### Predicted Sentiment: <span style='color:{color[sentiment]}'>{sentiment}</span>", unsafe_allow_html=True)

    st.subheader("Confidence Scores")
    for label, score in zip(labels, scores):
        st.write(f"{label}: {score:.4f}")
        st.progress(float(score))

    st.caption(f"Model: {MODEL_NAME}")
