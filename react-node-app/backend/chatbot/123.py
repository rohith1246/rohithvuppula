from flask import Flask, request, jsonify
import re
import string
import random
import json
from datetime import datetime
from nltk.chat.util import Chat, reflections
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from collections import defaultdict

app = Flask(__name__)

class NextopsonSupportBot:
    def __init__(self):
        self.train_data = [
            

            
            ('What is Nextopson?', 'Nextopson is a zero-brokerage real estate platform that directly connects property buyers and sellers. We make property transactions simple and cost-effective.'),
            ('How does Nextopson work?', 'Nextopson lets you list and find properties without any brokerage fees. Simply sign up, browse listings, or post your property to get started.'),
            ('Why choose Nextopson?', 'Nextopson offers zero brokerage, direct buyer-seller connection, verified listings, and a hassle-free property transaction experience.'),
            
            
            ('How do I list my property?', 'To list your property on Nextopson: 1) Sign up/Login 2) Click "Post Property" 3) Fill in property details 4) Upload photos 5) Submit for verification. Need help with any step?'),
            ('What details do I need to list?', 'You\'ll need to provide property location, type, size, price, amenities, and high-quality photos. Would you like a detailed listing guide?'),
            ('How long does listing take?', 'Property listing takes just 10-15 minutes. Our verification process typically completes within 24 hours.'),
            
            
            ('How to search properties?', 'Use our search filters to find properties by location, type, budget, and amenities. You can also save searches and get alerts for new matches.'),
            ('Can I save properties?', 'Yes! Create a free account to save favorite properties, set alerts, and track property updates.'),
            ('How to contact sellers?', 'Click "Contact Seller" on any listing to send a message or request property details directly through our platform.'),
            
            #
            ('What are the fees?', 'Nextopson is completely free! We charge zero brokerage and no hidden fees for listing or searching properties.'),
            ('Is there any commission?', 'No commission at all! Nextopson operates on a zero-brokerage model to make property transactions more affordable.'),
            ('Are there premium services?', 'All our core services are free. We may offer optional premium features for enhanced visibility in the future.'),
            
            
            ('Are listings verified?', 'Yes, our team verifies all property listings to ensure authenticity. We check property documents and seller credentials.'),
            ('Is it safe to use Nextopson?', 'Absolutely! We verify all listings, secure your data, and facilitate safe communication between buyers and sellers.'),
            ('How to report issues?', 'Use the "Report" button on listings or contact our support team through the help center for immediate assistance.'),
            
            
            ('How to create account?', 'Click "Sign Up" on nextopson.com, enter your details, verify your email, and start using our services immediately.'),
            ('Edit my listing', 'Log in to your account, go to "My Listings," select the property, and click "Edit" to update any information.'),
            ('Delete my listing', 'Access "My Listings" in your account, find the property, and use the "Delete Listing" option to remove it.'),
            
            
            ('Contact support', 'You can reach our support team through: 1) Help Center 2) support@nextopson.com 3) In-app chat 4) Customer care number.'),
            ('Technical issues', 'For technical issues, please try refreshing the page or clearing your browser cache. If the problem persists, contact our support team.'),
            ('Forgot password', 'Click "Forgot Password" on the login page, enter your registered email, and follow the reset instructions sent to you.'),
            
            
            ('Required documents', 'For listing: Property ownership proof, tax receipts, and ID proof. For buying: Just create an account to start viewing properties.'),
            ('Document verification', 'Our team verifies all property documents within 24 hours of submission to ensure authenticity.'),
            ('Update documents', 'Log in, go to "My Listings," select your property, and use the "Update Documents" option to add or modify documents.'),
            
            
            ('Find my listings', 'After logging in, click on "My Account" and select "My Listings" to view all your property listings.'),
            ('View saved properties', 'Access your saved properties through "My Account" → "Favorites" after logging in.'),
            ('Search filters', 'Use our advanced filters for location, property type, price range, amenities, and more to find your perfect property.'),
            
            
            ('How to buy property?', 'Browse listings, contact sellers directly, negotiate, and proceed with documentation. Our team can guide you through each step.'),
            ('Payment process', 'Payments are handled directly between buyers and sellers. We recommend secure payment methods and can provide guidance on the process.'),
            ('Property inspection', 'Schedule property visits directly with sellers through our platform. We recommend thorough inspection before proceeding.'),
        ]
        
        # Store conversation context
        self.conversations = defaultdict(list)
        
        # Store new learned responses
        self.learned_responses = []
        
        # Load learned data from file if exists
        self.load_learned_data()
        
        # Initialize the model
        self.initialize_model()
        
        # Confidence threshold for responses
        self.confidence_threshold = 0.4
        
        # Topic categories for context awareness
        self.topics = {
            'account': ['login', 'signup', 'password', 'profile', 'account'],
            'property_listing': ['list', 'sell', 'post', 'upload', 'property'],
            'property_search': ['search', 'find', 'browse', 'filter', 'properties'],
            'payment': ['fee', 'commission', 'payment', 'cost', 'price'],
            'support': ['help', 'contact', 'support', 'issue', 'problem']
        }

    def load_learned_data(self):
        """Load previously learned responses from file"""
        try:
            with open('learned_responses.json', 'r') as f:
                self.learned_responses = json.load(f)
                # Add learned responses to training data
                self.train_data.extend(self.learned_responses)
        except FileNotFoundError:
            pass

    def save_learned_data(self):
        """Save learned responses to file"""
        with open('learned_responses.json', 'w') as f:
            json.dump(self.learned_responses, f)

    def learn_new_response(self, question, answer):
        """Learn new question-answer pair"""
        new_pair = (question, answer)
        self.learned_responses.append(new_pair)
        self.train_data.append(new_pair)
        self.save_learned_data()
        self.initialize_model()

    def get_topic(self, text):
        """Identify the topic of the conversation"""
        text = text.lower()
        for topic, keywords in self.topics.items():
            if any(keyword in text for keyword in keywords):
                return topic
        return 'general'

    def maintain_context(self, session_id, user_input, response):
        """Maintain conversation context"""
        self.conversations[session_id].append({
            'user_input': user_input,
            'bot_response': response,
            'timestamp': datetime.now().isoformat(),
            'topic': self.get_topic(user_input)
        })
        
        # Keep only last 5 interactions for context
        if len(self.conversations[session_id]) > 5:
            self.conversations[session_id].pop(0)

    def get_response_with_context(self, session_id, user_input):
        """Get response considering conversation context"""
        if not isinstance(user_input, str) or not user_input.strip():
            return {
                'response': "I couldn't understand that. How can I help you with Nextopson's services?",
                'confidence': 0,
                'needs_learning': False
            }
        
        try:
            cleaned_input = self.preprocess_input(user_input)
            
            # Check for inappropriate language
            if self.contains_inappropriate_language(cleaned_input):
                return {
                    'response': "Let's keep our conversation professional. How can I assist you with your property needs?",
                    'confidence': 1,
                    'needs_learning': False
                }
            
            # Check predefined patterns
            for pattern, responses in self.pairs:
                if re.search(pattern, cleaned_input, re.IGNORECASE):
                    response = random.choice(responses)
                    self.maintain_context(session_id, user_input, response)
                    return {
                        'response': response,
                        'confidence': 1,
                        'needs_learning': False
                    }
            
            # Get context from previous conversations
            context = self.conversations.get(session_id, [])
            recent_topic = context[-1]['topic'] if context else None
            
            # Use ML model for prediction
            vectorized_input = [cleaned_input]
            try:
                # Get prediction probabilities
                proba = self.pipeline.predict_proba(vectorized_input)[0]
                max_proba = max(proba)
                
                if max_proba >= self.confidence_threshold:
                    predicted_response = self.pipeline.predict(vectorized_input)[0]
                    
                    # If the topic matches recent context, increase confidence
                    if recent_topic and recent_topic == self.get_topic(predicted_response):
                        max_proba += 0.1
                    
                    self.maintain_context(session_id, user_input, predicted_response)
                    return {
                        'response': predicted_response,
                        'confidence': float(max_proba),
                        'needs_learning': False
                    }
                else:
                    # Low confidence response
                    return {
                        'response': "I'm not entirely sure about this. Would you like me to learn the correct response?",
                        'confidence': float(max_proba),
                        'needs_learning': True
                    }
                
            except Exception as e:
                return {
                    'response': "I'm not sure about this. Would you like to teach me the correct response?",
                    'confidence': 0,
                    'needs_learning': True
                }
            
        except Exception as e:
            return {
                'response': "I'm having trouble understanding. Please try rephrasing your question about Nextopson's services.",
                'confidence': 0,
                'needs_learning': True
            }

    def preprocess_input(self, text):
        """Clean user input"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())

    def contains_inappropriate_language(self, text):
        """Check for inappropriate content"""
        inappropriate_words = set(['fuck', 'shit', 'damn', 'bitch', 'ass'])
        return any(word in text.split() for word in inappropriate_words)

    def initialize_model(self):
        """Initialize and train the model"""
        self.pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
        X_train = [x[0] for x in self.train_data]
        y_train = [x[1] for x in self.train_data]
        self.pipeline.fit(X_train, y_train)

# Initialize bot
support_bot = NextopsonSupportBot()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    response = support_bot.get_response_with_context(session_id, user_input)
    return jsonify(response)

@app.route('/learn', methods=['POST'])
def learn():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    
    if not question or not answer:
        return jsonify({'error': 'Both question and answer are required'}), 400
    
    support_bot.learn_new_response(question, answer)
    return jsonify({'message': 'Successfully learned new response'})

@app.route('/conversation_history', methods=['GET'])
def get_conversation_history():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    history = support_bot.conversations.get(session_id, [])
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)