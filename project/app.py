import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------
# Load Resources
# ---------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# reverse mapping index -> word
index_to_word = {}
for word, index in tokenizer.word_index.items():
    index_to_word[index] = word


# ---------------------------------
# Predict One Next Word
# ---------------------------------
def predict_next_word(text):
    text = text.lower()

    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    preds = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(preds)

    return index_to_word.get(predicted_index, "")


# ---------------------------------
# Generate Multiple Words
# ---------------------------------
def generate_text(seed_text, n_words):
    result = seed_text.lower()

    for _ in range(n_words):

        next_word = predict_next_word(result)

        if next_word == "":
            break

        result += " " + next_word

    return result


# ---------------------------------
# UI
# ---------------------------------
st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and generate multiple next words.")

user_input = st.text_input(
    "✍️ Enter text:",
    placeholder="Type a sentence here..."
)

num_words = st.slider(
    "🔢 Number of words to generate",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("🚀 Generate Text"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        output = generate_text(user_input, num_words)

        st.success("✅ Generated Text:")
        st.write(output)


st.markdown("---")
st.caption("LSTM-based Text Generation using Streamlit")