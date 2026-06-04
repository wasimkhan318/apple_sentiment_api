import os
import pickle
import string
import numpy as np
from flask import Flask, request, jsonify
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

with open(os.path.join(MODEL_DIR, "nb_model.pkl"), "rb") as f:
    nb_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "word_dic.pkl"), "rb") as f:
    word_dic = pickle.load(f)

with open(os.path.join(MODEL_DIR, "corpus.pkl"), "rb") as f:
    corpus = pickle.load(f)

STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)
LABEL_MAP = {1: "Negative", 3: "Neutral", 5: "Positive"}
tokenizer = TweetTokenizer()


def clean_tweet(text):
    tokens = tokenizer.tokenize(text)
    cleaned = []
    for token in tokens:
        token = token.lower()
        if token in STOP_WORDS:
            continue
        if all(ch in PUNCTUATION for ch in token):
            continue
        if "/" in token:
            continue
        cleaned.append(token)
    return cleaned


def vectorize(tokens):
    vector = np.zeros((1, len(word_dic)))
    for w in set(tokens):
        if w in word_dic:
            vector[0, word_dic[w]] = corpus.tf_idf(w, tokens)  # ← correct
    return scaler.transform(vector)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Apple Tweet Sentiment API",
        "version": "1.0.0",
        "description": "Predicts sentiment of Apple-related tweets using a Naive Bayes classifier.",
        "endpoints": {
            "GET /": "API info",
            "POST /predict": "Predict sentiment. Body: {\"tweet\": \"<text>\"}"
        },
        "labels": {"1": "Negative", "3": "Neutral", "5": "Positive"}
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None or "tweet" not in data:
        return jsonify({"error": "Request body must be JSON with a 'tweet' field."}), 400

    tweet_text = data["tweet"]

    if not isinstance(tweet_text, str) or tweet_text.strip() == "":
        return jsonify({"error": "The 'tweet' field must be a non-empty string."}), 400

    tokens = clean_tweet(tweet_text)

    if len(tokens) == 0:
        return jsonify({"error": "No meaningful words found after cleaning the tweet."}), 422

    vector = vectorize(tokens)
    vector_scaled = scaler.transform(vector)

    prediction = nb_model.predict(vector_scaled)[0]
    probabilities = nb_model.predict_proba(vector_scaled)[0]
    confidence = float(np.max(probabilities))

    label = int(prediction)
    sentiment = LABEL_MAP.get(label, "Unknown")

    return jsonify({
        "tweet": tweet_text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "label": label
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4200))
    app.run(host="0.0.0.0", port=port, debug=False)