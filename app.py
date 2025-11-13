from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = tf.keras.models.load_model("spam_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']

    # Preprocess input
    seq = tokenizer.texts_to_sequences([email_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Prediction
    prob = model.predict(padded)[0][0]
    prob_percent = f"{prob * 100:.2f}"
    result = "SPAM 🚫" if prob > 0.4 else "NOT SPAM ✅"

    return render_template(
        'index.html',
        prediction=result,
        prob=f"{prob_percent}%",
        email_text=email_text
    )

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)