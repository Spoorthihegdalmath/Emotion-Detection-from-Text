from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up NLTK resources in case they aren't downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load the model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
VEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'vectorizer.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print("Model not found. Please run train_model.py first.")
    model = None

try:
    with open(VEC_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Vectorizer loaded successfully from {VEC_PATH}")
except FileNotFoundError:
    print("Vectorizer not found. Please run train_model.py first.")
    vectorizer = None

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer is not loaded on the server."}), 500
        
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided in the request body."}), 400
        
    user_text = data['text']
    
    if not user_text.strip():
        return jsonify({"error": "Please enter some text."}), 400

    cleaned_text = preprocess_text(user_text)
    
    if not cleaned_text.strip():
        # Fallback to prevent crashing if text entirely consists of stopwords/punctuation
        cleaned_text = user_text # let the vectorizer handle it as best it can
        # but realistically, vectorizer might output all 0s.

    vec_text = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vec_text)[0]
    
    # Get confidence scores
    try:
        probabilities = model.predict_proba(vec_text)[0]
        classes = model.classes_
        confidences = {str(cls): round(float(prob) * 100, 2) for cls, prob in zip(classes, probabilities)}
    except AttributeError:
        # Fallback if model doesn't support predict_proba
        confidences = {prediction: 100.0}

    # Format emotion output nicely
    emotion_map = {
        'happy': 'Happy 😊',
        'sad': 'Sad 😢',
        'angry': 'Angry 😠',
        'fear': 'Fear 😨',
        'surprise': 'Surprise 😲',
        'neutral': 'Neutral 😐'
    }
    
    formatted_emotion = emotion_map.get(prediction, prediction.capitalize())
    top_confidence = f"{confidences.get(prediction, 0)} %"
    
    return jsonify({
        "emotion": formatted_emotion,
        "confidence": top_confidence,
        "raw_emotion": prediction,
        "all_confidences": confidences
    })

if __name__ == '__main__':
    app.run(debug=True)
