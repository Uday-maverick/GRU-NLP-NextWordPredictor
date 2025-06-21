import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

# Load the LSTM model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "🤔 (no suggestion)"

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0f0f0f;
        color: #ffffff;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #5c5c5c;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 25px;
        padding: 0.5em 1.5em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1c1c;
    }
    </style>
""", unsafe_allow_html=True)

# App layout
st.markdown("<h1 style='text-align: center;'>🚀 Next Word Predictor using LSTM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Type a sequence and let AI guess the next word 🔮</p>", unsafe_allow_html=True)

# User input
input_text = st.text_input("Type a sequence of words:", "")

# Predict button
if st.button("🔍 Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.markdown(f"<h3 style='text-align: center;'>🧠 Predicted Next Word: <span style='color: #00ffff'>{next_word}</span></h3>", unsafe_allow_html=True)
