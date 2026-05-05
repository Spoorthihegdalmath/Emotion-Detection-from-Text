from transformers import pipeline
import torch
import warnings
import urllib3
import re

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Upgraded to 28 advanced emotions
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# Macro Sentiment classification
MACRO_MAPPING = {
    "positive": ["admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness", "urgency"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise", "questioning"],
    "neutral": ["neutral"]
}

EMPATHY_RESPONSES = {
    # High energy positive
    "joy": "That's absolutely wonderful! I'm thrilled to hear that.",
    "excitement": "That sounds incredibly exciting! Tell me more.",
    "love": "That's so beautiful! Cherish those feelings.",
    "pride": "You should be proud! That's a great achievement.",
    "amusement": "Haha, that's genuinely funny!",
    
    # Low energy positive
    "caring": "That's very sweet and considerate of you.",
    "gratitude": "It's always great to appreciate the good things.",
    "optimism": "Love the positive outlook! Keep it up.",
    "relief": "Phew! Sounds like a weight off your shoulders.",
    
    # High energy negative
    "anger": "I can sense your frustration. It's completely understandable to feel this way.",
    "annoyance": "That sounds quite bothersome. I totally get why you're annoyed.",
    "disgust": "Yikes. That sounds genuinely unpleasant.",
    "fear": "It's okay to be scared. Take a deep breath.",
    
    # Low energy negative
    "sadness": "I'm really sorry you're feeling this way. I'm here for you.",
    "grief": "I am deeply sorry for your loss and pain.",
    "disappointment": "That's definitely let-down. I understand why you feel that way.",
    "remorse": "We all make mistakes. Don't be too hard on yourself.",
    "embarrassment": "Oh no! But don't worry, everyone has those moments.",
    "nervousness": "It's completely normal to feel anxious. You've got this!",
    
    # Ambiguous
    "surprise": "Wow, what a surprising turn of events!",
    "curiosity": "That's an interesting thought! Let's explore that.",
    "confusion": "It sounds a bit perplexing. Let's try to untangle it together.",
    "realization": "Ah, an 'aha!' moment. Makes sense now, doesn't it?",
    "questioning": "That's a very good question. Let's think about it.",
    "urgency": "I hear the urgency. Let's tackle this quickly.",
    
    # Defaults
    "positive": "That sounds quite positive!",
    "negative": "I understand this might be challenging.",
    "ambiguous": "That's quite interesting.",
    "neutral": "I'm listening. Please go on."
}

class AdvancedEmotionModel:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading transformer model {MODEL_NAME} on device {self.device}...")
        self.classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, device=self.device)
        print("Transformer loaded.")

    def get_intensity(self, max_score: float) -> str:
        if max_score > 0.8:
            return "High"
        elif max_score > 0.5:
            return "Medium"
        return "Low"

    def get_macro_sentiment(self, primary_emotion: str) -> str:
        for macro, emotions in MACRO_MAPPING.items():
            if primary_emotion in emotions:
                return macro.capitalize()
        return "Neutral"

    def get_empathy_response(self, primary_emotion: str) -> str:
        if primary_emotion in EMPATHY_RESPONSES:
            return EMPATHY_RESPONSES[primary_emotion]
        
        # Fallback to macro mapping
        macro = self.get_macro_sentiment(primary_emotion).lower()
        return EMPATHY_RESPONSES.get(macro, EMPATHY_RESPONSES["neutral"])

    def predict(self, text: str) -> dict:
        results = self.classifier(text)[0]
        
        confidences = {res['label']: round(res['score'] * 100, 2) for res in results}
        
        # Sort to find highest overall
        sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)
        primary = sorted_emotions[0]['label']
        max_score = sorted_emotions[0]['score']
        
        emotions_set = set([primary])
        
        # Heuristic rules for explicit expressions
        lower_text = text.lower()
        if '?' in text or lower_text.startswith(('what', 'why', 'how', 'when', 'who', 'where')):
            emotions_set.add('questioning')
            if primary == 'neutral':
                primary = 'questioning'
            if 'questioning' not in confidences:
                confidences['questioning'] = 85.0
                
        if '!' in text and any(w in lower_text for w in ['now', 'quick', 'hurry', 'fast', 'urgent', 'immediately']):
            emotions_set.add('urgency')
            if 'urgency' not in confidences:
                confidences['urgency'] = 90.0
        
        # Check full sentence tail probabilities
        for res in sorted_emotions:
            if res['score'] >= 0.15:
                emotions_set.add(res['label'])
                
        # NLP Clause Chunking for explicit Mixed Emotions
        clauses = re.split(r'\bbut\b|\band\b|\bhigher\b|\balso\b|,|\.|;|!', text.lower())
        for clause in clauses:
            clause = clause.strip()
            if len(clause.split()) >= 3:
                clause_res = self.classifier(clause)[0]
                top_clause = max(clause_res, key=lambda x: x['score'])
                
                if top_clause['score'] > 0.4 and top_clause['label'] != 'neutral':
                    emotions_set.add(top_clause['label'])
                    
        emotions = list(emotions_set)
        intensity = self.get_intensity(max_score)
        macro_sentiment = self.get_macro_sentiment(primary)
        empathy_response = self.get_empathy_response(primary)
        
        return {
            "emotions": emotions,
            "confidences": confidences,
            "primary_emotion": primary,
            "intensity": intensity,
            "macro_sentiment": macro_sentiment,
            "empathy_response": empathy_response
        }

    def explain(self, text: str) -> dict:
        """
        Lightweight Ablation-based Explainability
        """
        words = text.split()
        if len(words) <= 1:
             return {words[0]: 1.0} if words else {}
             
        base_pred = self.classifier(text)[0]
        primary_label = max(base_pred, key=lambda x: x['score'])['label']
        base_score = max(base_pred, key=lambda x: x['score'])['score']
        
        importance = {}
        for i, word in enumerate(words):
            masked_text = " ".join(words[:i] + words[i+1:])
            if not masked_text.strip():
                continue
            
            new_pred = self.classifier(masked_text)[0]
            new_score = next((x['score'] for x in new_pred if x['label'] == primary_label), 0)
            
            drop = max(0, base_score - new_score)
            importance[word] = round(drop * 100, 2)
            
        return importance
