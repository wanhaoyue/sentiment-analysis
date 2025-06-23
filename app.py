import streamlit as st
import joblib
import spacy
import os
import urllib.request

# URL of your hosted .pkl file (for example, from Google Drive or Dropbox)
MODEL_URL = "https://drive.google.com/file/d/1oc5v81ClYCImgJl4dIDUVkeeULZTTtuZ/view?usp=drive_link"

# Download model if not already downloaded
if not os.path.exists("sentiment_rf_pipeline.pkl"):
    urllib.request.urlretrieve(MODEL_URL, "sentiment_rf_pipeline.pkl")

# Load the model
model = joblib.load("sentiment_rf_pipeline.pkl")



# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load trained model pipeline
model = joblib.load("sentiment_rf_pipeline.pkl")

# Preprocessing function
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

# Class labels
classes = ['Irrelevant', 'Natural', 'Negative', 'Positive']

# Streamlit UI
st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet or text below to analyze its sentiment:")

# User input
user_input = st.text_area("Tweet Text", height=150)

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        processed_input = [preprocess(user_input)]

        # Predict
        pred = model.predict(processed_input)
        predicted_label = classes[pred[0]]

        # Show result
        st.success(f"Predicted Sentiment: **{predicted_label}**")
