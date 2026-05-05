# Advanced Emotion AI Platform 🧠

This project is an **Industry-Grade Full-Stack Application** designed to detect emotional subtext and sentiment intensity from unstructured chat messages and direct voice input. 

Instead of basic keyword matching, it leverages **Deep Learning Transformers** and **Advanced NLP Clause Chunking** to accurately dissect complex, multi-emotion sentences in real-time.

## ✨ Key Features
- **Deep Learning Embeddings (Transformers)**: Utilizes HuggingFace's `distilroberta-base-emotion` model running natively via PyTorch. It maps incoming sentences to 7 explicit emotion labels (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral).
- **Multi-Label NLP Clause Chunking**: Dynamically splits complex sentences separated by conjunctions (e.g., "but", "and") into logical clauses. It detects if a single sentence packs multiple contrasting sentiments—returning an array of emotions alongside an explicit **Intensity Level** (Low/Medium/High).
- **Model Explainability Engine**: The backed uses Ablation testing (similar to LIME/SHAP). It blacks out portions of your sentence to mathematically prove *which specific words* caused the AI to predict the resulting emotion.
- **Voice-to-Text Dictation**: Native browser `Web Speech API` integration captures your microphone input remotely for instantaneous analysis locally.
- **React UI & Real-Time Dashboards**: A gorgeous, Glassmorphic Chatbot interface built with Vite + React. Features a live `Chart.js` dashboard that graphs pie distributions and sentiment timelines over your chat history.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
Open your terminal and clone the project from GitHub:
```bash
git clone https://github.com/YourUsername/Emotion-AI-Platform.git
cd Emotion-AI-Platform
```
*(Replace `YourUsername/Emotion-AI-Platform.git` with your actual GitHub repository URL once uploaded).*

### 2. Boot the Deep Learning Backend
The backend runs on **FastAPI**. It requires Python installed on your system.
```bash
cd backend
pip install -r requirements.txt
pip install "numpy<2"  # Essential for avoiding PyTorch/Transformers clashes on new NumPy versions
python app.py
```
> **Note:** The first time you run this, it will download the ~330MB Transformer model weights. Wait until the terminal says: `Application startup complete` (Port `http://localhost:8000`).

### 3. Boot the React Frontend
Open a **new separate terminal tab**, and initialize the Vite configuration:
```bash
cd frontend
npm install
npm run dev
```
> The React Dashboard will hot-load onto your browser, typically at `http://localhost:5173`. Open the link, allow microphone access, and start chatting!

---

## 📡 API Specification Layer
FastAPI auto-generates a Swagger UI. Navigate to `http://localhost:8000/docs` while the backend runs to experiment directly with the POST payload endpoints:
- `POST /predict`: Submit `{"text": "your sentence"}` to receive multi-class emotion strings and intensities.
- `POST /explain`: Submit `{"text": "your sentence"}` to receive a mapped dictionary quantifying every single word's mathematical importance against the emotional prediction.
