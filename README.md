# 💬 Twitter Sentiment Analysis App

This is a simple web app to analyze the **sentiment of tweets** using the powerful Transformer model [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

The app is built with **Streamlit** and is deployed on **Streamlit Cloud** — no setup needed, just open the URL!

## 🚀 Features

- Enter any tweet or short sentence and get sentiment analysis.
- Supports 3 sentiment classes:
  - Positive 😀
  - Neutral 😐
  - Negative 😞
- Shows prediction confidence scores.
- Example tweets provided to try out.

## 🛠️ Technologies Used

- [Transformers](https://huggingface.co/docs/transformers/index)
- [Streamlit](https://streamlit.io/)
- [Torch](https://pytorch.org/)

## 🌐 How to Run Locally
```bash
git clone https://github.com/YOURUSERNAME/twitter-sentiment-streamlit.git
cd twitter-sentiment-streamlit
pip install -r requirements.txt
streamlit run app.py
