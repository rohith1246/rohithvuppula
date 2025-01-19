from flask import Flask, request, jsonify
import re
import string
import random
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import os

app = Flask(__name__)

class NextopsonSupportBot:
    def __init__(self):
        self.train_data = [
            ('What is Nextopson?', 'Nextopson is a zero-brokerage real estate platform that directly connects property buyers and sellers. We make property transactions simple and cost-effective.'),
            ('How does Nextopson work?', 'Nextopson lets you list and find properties without any brokerage fees. Simply sign up, browse listings, or post your property to get started.'),
            # ... (rest of your training data)
        ]
        
        self.pairs = [
            ('hello|hi|hey', [
                'Welcome to Nextopson support! How can I assist you today?',
                'Hello! How may I help you with your property needs?',
                'Hi! Looking to buy, sell, or need help with Nextopson?'
            ]),
            # ... (rest of your pairs)
        ]
        
        self.initialize_model()

    def get_response(self, user_input):
        if not isinstance(user_input, str) or not user_input.strip():
            return {
                "response": "I couldn't understand that. How can I help you with Nextopson's services?",
                "confidence": 0.0
            }
        
        try:
            cleaned_input = self.preprocess_input(user_input)
            
            if self.contains_inappropriate_language(cleaned_input):
                return {
                    "response": "Let's keep our conversation professional. How can I assist you with your property needs?",
                    "confidence": 1.0
                }
            
            for pattern, responses in self.pairs:
                if re.search(pattern, cleaned_input, re.IGNORECASE):
                    return {
                        "response": random.choice(responses),
                        "confidence": 1.0
                    }
            
            vectorized_input = [cleaned_input]
            try:
                prediction = self.pipeline.predict(vectorized_input)[0]
                confidence = max(self.pipeline.predict_proba(vectorized_input)[0])
                return {
                    "response": prediction,
                    "confidence": float(confidence)
                }
            except Exception:
                return {
                    "response": "Could you please provide more details about your query? I'm here to help with all Nextopson-related questions.",
                    "confidence": 0.0
                }
            
        except Exception as e:
            return {
                "response": "I'm having trouble understanding. Please try rephrasing your question about Nextopson's services.",
                "confidence": 0.0
            }

    def preprocess_input(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())

    def contains_inappropriate_language(self, text):
        inappropriate_words = set(['fuck', 'shit', 'damn', 'bitch', 'ass'])
        return any(word in text.split() for word in inappropriate_words)

    def initialize_model(self):
        self.pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
        X_train = [x[0] for x in self.train_data]
        y_train = [x[1] for x in self.train_data]
        self.pipeline.fit(X_train, y_train)
        
        # Save the model
        with open('chatbot_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)

# Initialize bot
support_bot = NextopsonSupportBot()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "error": "No message provided"
            }), 400
            
        response = support_bot.get_response(data['message'])
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)