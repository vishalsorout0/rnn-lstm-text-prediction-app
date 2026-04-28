# rnn-lstm-text-prediction-app
A deep learning NLP project using RNN/LSTM for next word prediction with Streamlit deployment. Includes preprocessing, tokenization, model training, and interactive text generation UI.


# 🧠 LSTM Next Word Prediction using Streamlit

This project is a Deep Learning based Natural Language Processing (NLP) application that predicts the next word in a sentence using an LSTM (Long Short-Term Memory) neural network.

The model was trained on a quote/text dataset and deployed using Streamlit for an interactive web interface.

---

## 🚀 Project Demo

Enter a sentence such as:

the meaning of life is

Output:

the meaning of life is to live with purpose

(Generated output depends on training data quality and model weights.)

---

## 📌 Features

✅ Next word prediction using LSTM  
✅ Multi-word text generation  
✅ Text preprocessing pipeline  
✅ Tokenization and sequence creation  
✅ Padding for fixed-length inputs  
✅ Trained TensorFlow / Keras model  
✅ Streamlit web app deployment  
✅ Clean and interactive UI

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Streamlit
- Pickle

---

## 📂 Project Structure

```bash
.
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── qoute_dataset.csv
├── notebook.ipynb
└── README.md
---
🧪 Model Workflow
1. Data Preprocessing
Converted text to lowercase
Removed punctuation
Cleaned text data
2. Tokenization

Used Keras Tokenizer to convert words into integer tokens.

3. Sequence Generation

Example:

Input:

life is

Target:

beautiful

Created multiple training sequences from sentences.

4. Padding

Used pre-padding so all input sequences have equal length.

5. Training

Used LSTM architecture:

Embedding Layer
LSTM Layer
Dense Softmax Output Layer
6. Prediction

Given seed text:

what are you

Model predicts next likely word.
RNNs for sequential text tasks because they can retain long-term dependencies and reduce vanishing gradient problems.
---
📈 Limitations
Small dataset (~3000 rows)
Limited vocabulary coverage
Predictions may be inaccurate for unseen phrases

This project is focused on learning and implementation of the complete NLP workflow.
---
🔮 Future Improvements
Larger dataset
Better hyperparameter tuning
Stacked LSTM / GRU
Attention mechanism
Transformer-based model
Better UI/UX
Cloud deployment
▶️ Run Locally
Install dependencies
pip install -r requirements.txt
Start Streamlit app
python -m streamlit run app.py
--- 
📸 Screenshots


📚 Learning Outcomes

Through this project I learned:

NLP preprocessing
Sequence modeling
Deep learning for text
Model saving/loading
Deployment using Streamlit
Debugging real-world issues
