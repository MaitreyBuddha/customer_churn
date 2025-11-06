## Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Loading the word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

## Load the trained model
model = load_model("movie_review_rnn_model.h5")


## Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preprocess_review(review):
    # Tokenize the review
    tokens = review.lower().split()
    # Convert words to their corresponding indices
    encoded_review = [word_index.get(word, 2) + 3 for word in tokens]  # 2 is for unknown words
    # Pad the sequence
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_review_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]



import streamlit as st

st.title("IMDB Movie Review Sentiment Classification")
st.write("Enter a movie review below to predict its sentiment (positive or negative).")

## User Input
user_review = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.write("Please enter a valid review.")
    else:
        sentiment, confidence = predict_review_sentiment(user_review)
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence: **{confidence:.2f}**")