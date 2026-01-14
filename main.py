import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

## Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for (key,value) in word_index.items()}

##Load the saved model
model = load_model("simple_rnn_imdb.h5")

## Helper function
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

##Function to preprocess user input
def preprocess_input(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

###Prediction function
def predict_review(review):
    processed_review = preprocess_input(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

## Design my streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment.")
user_input = st.text_area("Movie Review:", height=200)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.write("Please enter a valid movie review.")
    else:
        sentiment, score = predict_review(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}** (Score: {score:.4f})")