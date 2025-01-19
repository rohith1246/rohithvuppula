import re
import string
import random
import json
import logging
from datetime import datetime
from nltk.chat.util import Chat, reflections
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from collections import defaultdict
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NextopsonSupportBot:
    def __init__(self):
        # Initialize conversation memory
        self.conversation_memory = defaultdict(list)
        self.memory_size = 5
        
        # Sentiment patterns
        self.sentiment_patterns = {
            'positive': r'\b(great|awesome|excellent|good|wonderful|fantastic|thank|thanks|helpful)\b',
            'negative': r'\b(bad|poor|terrible|awful|horrible|worst|useless|waste|frustrated)\b',
            'urgent': r'\b(urgent|asap|emergency|immediately|quick|hurry)\b'
        }
        
        # Property type patterns
        self.property_types = {
            'residential': r'\b(house|apartment|flat|villa|condo|residential|1bhk|2bhk|3bhk)\b',
            'commercial': r'\b(office|shop|retail|commercial|warehouse|store)\b',
            'land': r'\b(plot|land|acre|vacant|empty)\b'
        }
        
        # Location and price patterns
        self.location_patterns = r'\b(near|location|area|city|locality|address)\b'
        self.price_patterns = r'\b(price|cost|budget|expensive|cheap|affordable)\b'

        self.train_data = [
            # Basic Information
            ('What is Nextopson?', 'Nextopson is a zero-brokerage real estate platform that directly connects property buyers and sellers. We make property transactions simple and cost-effective.'),
            ('How does Nextopson work?', 'Nextopson lets you list and find properties without any brokerage fees. Simply sign up, browse listings, or post your property to get started.'),
            ('Why choose Nextopson?', 'Nextopson offers zero brokerage, direct buyer-seller connection, verified listings, and a hassle-free property transaction experience.'),
            
            # Property Listing
            ('How do I list my property?', 'To list your property on Nextopson: 1) Sign up/Login 2) Click "Post Property" 3) Fill in property details 4) Upload photos 5) Submit for verification. Need help with any step?'),
            ('What details do I need to list?', 'You\'ll need to provide property location, type, size, price, amenities, and high-quality photos. Would you like a detailed listing guide?'),
            ('How long does listing take?', 'Property listing takes just 10-15 minutes. Our verification process typically completes within 24 hours.'),
            
            # Property Search
            ('How to search properties?', 'Use our search filters to find properties by location, type, budget, and amenities. You can also save searches and get alerts for new matches.'),
            ('Can I save properties?', 'Yes! Create a free account to save favorite properties, set alerts, and track property updates.'),
            ('How to contact sellers?', 'Click "Contact Seller" on any listing to send a message or request property details directly through our platform.'),
            
            # Fees and Pricing
            ('What are the fees?', 'Nextopson is completely free! We charge zero brokerage and no hidden fees for listing or searching properties.'),
            ('Is there any commission?', 'No commission at all! Nextopson operates on a zero-brokerage model to make property transactions more affordable.'),
            ('Are there premium services?', 'All our core services are free. We may offer optional premium features for enhanced visibility in the future.'),
            
            # Safety and Verification
            ('Are listings verified?', 'Yes, our team verifies all property listings to ensure authenticity. We check property documents and seller credentials.'),
            ('Is it safe to use Nextopson?', 'Absolutely! We verify all listings, secure your data, and facilitate safe communication between buyers and sellers.'),
            ('How to report issues?', 'Use the "Report" button on listings or contact our support team through the help center for immediate assistance.'),
            
            # Account Management
            ('How to create account?', 'Click "Sign Up" on nextopson.com, enter your details, verify your email, and start using our services immediately.'),
            ('Edit my listing', 'Log in to your account, go to "My Listings," select the property, and click "Edit" to update any information.'),
            ('Delete my listing', 'Access "My Listings" in your account, find the property, and use the "Delete Listing" option to remove it.'),
            
            # Support
            ('Contact support', 'You can reach our support team through: 1) Help Center 2) support@nextopson.com 3) In-app chat 4) Customer care number.'),
            ('Technical issues', 'For technical issues, please try refreshing the page or clearing your browser cache. If the problem persists, contact our support team.'),
            ('Forgot password', 'Click "Forgot Password" on the login page, enter your registered email, and follow the reset instructions sent to you.'),
            
            # Documents
            ('Required documents', 'For listing: Property ownership proof, tax receipts, and ID proof. For buying: Just create an account to start viewing properties.'),
            ('Document verification', 'Our team verifies all property documents within 24 hours of submission to ensure authenticity.'),
            ('Update documents', 'Log in, go to "My Listings," select your property, and use the "Update Documents" option to add or modify documents.'),
            
            # Navigation
            ('Find my listings', 'After logging in, click on "My Account" and select "My Listings" to view all your property listings.'),
            ('View saved properties', 'Access your saved properties through "My Account" → "Favorites" after logging in.'),
            ('Search filters', 'Use our advanced filters for location, property type, price range, amenities, and more to find your perfect property.'),
            
            # Property Transaction
            ('How to buy property?', 'Browse listings, contact sellers directly, negotiate, and proceed with documentation. Our team can guide you through each step.'),
            ('Payment process', 'Payments are handled directly between buyers and sellers. We recommend secure payment methods and can provide guidance on the process.'),
            ('Property inspection', 'Schedule property visits directly with sellers through our platform. We recommend thorough inspection before proceeding.'),

            # New Additional Training Data
            
            # Mobile App
            ('Is there a mobile app?', 'Yes, Nextopson is available on both iOS and Android. Download our app to search properties and manage listings on the go.'),
            ('App features', 'Our mobile app offers property search, instant notifications, chat with sellers/buyers, and easy listing management.'),
            ('App not working', 'Try updating the app to the latest version, check your internet connection, or clear the app cache. Contact support if issues persist.'),
            
            # Virtual Services
            ('Virtual tour', 'Many properties offer virtual tours. Look for the "360° View" icon on listings to explore properties virtually.'),
            ('Online documentation', 'You can upload and verify documents online through our secure platform. We accept digital signatures for most documents.'),
            ('Video calling', 'Use our built-in video calling feature to have virtual meetings with property owners or buyers.'),
            
            # Property Types
            ('Types of properties', 'We list residential properties (apartments, houses, villas), commercial spaces (offices, shops), and land/plots.'),
            ('Residential options', 'Browse apartments, independent houses, villas, penthouses, studio apartments, and more in our residential section.'),
            ('Commercial properties', 'Find offices, retail spaces, warehouses, industrial properties, and commercial land in our commercial section.'),
            
            # Location Based
            ('Popular locations', 'View trending localities, upcoming areas, and premium locations in your city with our location guides.'),
            ('Nearby amenities', 'Each listing shows nearby schools, hospitals, markets, and public transport options within a 5km radius.'),
            ('Area guides', 'Access detailed area guides with information about locality, infrastructure, prices, and future development plans.'),
            
            # Pricing and Loans
            ('Price negotiation', 'You can negotiate directly with sellers through our platform. We provide price trends to help make informed decisions.'),
            ('Home loans', 'Compare home loan offers from multiple banks through our platform. We have partnered with leading financial institutions.'),
            ('EMI calculator', 'Use our EMI calculator to estimate monthly payments based on loan amount, interest rate, and tenure.'),
            
            # Legal
            ('Legal verification', 'We help verify property legal status and documentation. Optional legal assistance is available through our partner lawyers.'),
            ('Property ownership', 'We verify property ownership and ensure all listings have clear titles before they go live on our platform.'),
            ('Legal documents', 'Get guidance on required legal documents like sale deed, property tax receipts, NOC, and occupancy certificate.'),
            
            # Premium Features
            ('Featured listing', 'Boost your property visibility with our featured listing option. Your property appears at the top of search results.'),
            ('Premium membership', 'Premium members get priority support, advanced analytics, and exclusive access to pre-launch properties.'),
            ('Marketing services', 'We offer professional photography, 3D tours, and social media promotion for premium listings.'),
            
            # Rental Properties
            ('Rental listing', 'List your property for rent with detailed terms, preferred tenant profile, and rental agreement requirements.'),
            ('Tenant verification', 'We offer tenant verification services including background checks and document verification.'),
            ('Rental agreement', 'Access standard rental agreement templates or get customized agreements through our legal partners.'),
            
            # Investment
            ('Investment advice', 'Our market insights and property analytics help you make informed investment decisions.'),
            ('ROI calculator', 'Calculate potential returns on your property investment using our ROI calculator tool.'),
            ('Market trends', 'Access real-time market trends, price history, and future projections for different localities.'),
            
            # Additional Services
            ('Interior design', 'Connect with our partner interior designers for home renovation and decoration services.'),
            ('Packers and movers', 'Book verified packers and movers through our platform for hassle-free relocation.'),
            ('Property management', 'Our property management services help you maintain and manage your property remotely.'),
            
            # Support Queries
            ('Response time', 'We typically respond to queries within 2 hours during business hours (9 AM - 6 PM).'),
            ('Feedback', 'Share your feedback through our app/website or email us at feedback@nextopson.com'),
            ('File complaint', 'Report issues or file complaints through our grievance redressal system for quick resolution.')
        ]

 


        # Chat patterns
        self.pairs = [
            (r'\b(hello|hi|hey|hii|hiiii|hlo|howdy|greetings)\b', [
                'Welcome to Nextopson support! How can I assist you today?',
                'Hello! How may I help you with your property needs?',
                'Hi! Looking to buy, sell, or need help with Nextopson?',
                'Greetings! How can I make your property journey easier today?'
            ]),
            (r'\b(bye|goodbye|see you|cya|good night|good morning)\b', [
                'Thank you for choosing Nextopson! Feel free to return if you need more assistance.',
                'Have a great day! We\'re here 24/7 for your property needs.',
                'Goodbye! Don\'t hesitate to contact us for any property-related help.'
            ]),
             (r'\b(what is|tell me about|about|explain) nextopson\b', [
                'Nextopson is a zero-brokerage real estate platform that directly connects property buyers and sellers.',
                'We are a platform making property transactions simple and cost-effective with zero brokerage fees.',
                'Nextopson is your go-to platform for direct property deals without any brokerage charges.'
            ]),
            (r'\b(how to|how do I|can I|help with) (list|post|add|sell) (my |a )?property\b', [
                'To list your property: 1) Sign up/Login 2) Click "Post Property" 3) Fill details 4) Upload photos 5) Submit for verification.',
                'Listing your property is easy! Just login, click Post Property, and follow our simple steps.',
                'Ready to list? Login and use our simple property posting process - takes just 10-15 minutes!'
            ]),
            (r'\b(how to|can I|help|want to) (search|find|look for|browse) properties\b', [
                'Use our search filters to find properties by location, type, budget, and amenities. Save searches for updates!',
                'Browse properties using our advanced filters - set your preferences and find your perfect match.',
                'Finding properties is easy! Set your criteria and use our smart search filters.'
            ]),
            (r'\b(what are the|any|how much) (fees|charges|commission|cost)\b', [
                'Nextopson is completely free! We charge zero brokerage and no hidden fees.',
                'Good news - we have no fees or commission! Our platform is completely free to use.',
                'Zero brokerage, zero commission, zero hidden charges - that\'s our promise!'
            ]),
            
            # Account Related
            (r'\b(how to|can I|help with) (create|make|get) (an )?account\b', [
                'Click "Sign Up" on nextopson.com, enter your details, verify email, and start using our services!',
                'Creating an account is quick and easy - just sign up with your email and get started.',
                'Join Nextopson in minutes! Click Sign Up and follow our simple registration process.'
            ]),
            (r'\b(how to|can I|need to) (contact|reach|get) support\b', [
                'Reach our support team via: 1) Help Center 2) support@nextopson.com 3) In-app chat 4) Customer care.',
                'Need help? Contact us through our Help Center or email support@nextopson.com',
                'Our support team is available 24/7 through multiple channels - choose what works best for you!'
            ]),
            
           
            (r'\b(what|which) documents (do I need|are required|needed)\b', [
                'For listing: Property ownership proof, tax receipts, and ID proof. For buying: Just create an account!',
                'Required documents include property ownership proof and basic ID verification.',
                'Keep ready: property documents, ID proof, and tax receipts for quick verification.'
            ]),
            
            
            (r'\b(is it|how) safe|security|verification\b', [
                'Absolutely! We verify all listings, secure your data, and ensure safe buyer-seller communication.',
                'Your safety is our priority! We verify all properties and users for secure transactions.',
                'We maintain strict verification processes for all listings and users to ensure safety.'
            ]),
           
            
          
            (r'\b(what|which) (type|kinds|sorts) of properties\b', [
                'We list residential (apartments, houses, villas) and commercial properties (offices, shops), plus land/plots.',
                'Browse various property types: residential, commercial, land, and more!',
                'Find everything from apartments to commercial spaces on our platform.'
            ]),
             
             

            (r'\b(thanks|thank you|thx|thankyou|appreciate)\b', [
                'You\'re welcome! Let us know if you need anything else.',
                'Happy to help! Feel free to reach out for any property-related queries.',
                'Glad to assist! Don\'t hesitate to ask if you have more questions.'
            ])
        ]

        # Initialize model and chat
        self.initialize_model()
        self.chat = Chat(self.pairs, reflections)

    def initialize_model(self):
        """Initialize and train the enhanced model"""
        self.pipeline = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000,
                stop_words='english',
                min_df=2,
                max_df=0.95
            ),
            MultinomialNB(alpha=0.1)
        )
        
        # Prepare and preprocess training data
        X_train = [self.preprocess_input(x[0]) for x in self.train_data]
        y_train = [x[1] for x in self.train_data]
        
        try:
            self.pipeline.fit(X_train, y_train)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise

    def preprocess_input(self, text):
        """Enhanced input preprocessing"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep question marks and basic punctuation
        text = re.sub(r'[^\w\s?.!,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize property-related terms
        property_terms = {
            'apt': 'apartment',
            'prop': 'property',
            'comm': 'commercial',
            'resi': 'residential',
            'loc': 'location',
            '1bhk': 'one bhk',
            '2bhk': 'two bhk',
            '3bhk': 'three bhk'
        }
        
        words = text.split()
        normalized_words = [property_terms.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def analyze_input(self, text):
        """Analyze user input for patterns and context"""
        analysis = {
            'sentiment': self.detect_sentiment(text),
            'property_type': self.detect_property_type(text),
            'has_location': bool(re.search(self.location_patterns, text, re.IGNORECASE)),
            'has_price': bool(re.search(self.price_patterns, text, re.IGNORECASE)),
            'word_count': len(text.split()),
            'is_question': '?' in text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return analysis

    def detect_sentiment(self, text):
        """Detect sentiment in user input"""
        for sentiment, pattern in self.sentiment_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return sentiment
        return 'neutral'

    def detect_property_type(self, text):
        """Detect property type from user input"""
        for prop_type, pattern in self.property_types.items():
            if re.search(pattern, text, re.IGNORECASE):
                return prop_type
        return 'general'

    def contains_inappropriate_language(self, text):
        """Check for inappropriate content"""
        inappropriate_words = set([
            'fuck', 'shit', 'damn', 'bitch', 'ass',
            'crap', 'stupid', 'idiot', 'dumb'
        ])
        words = text.lower().split()
        return any(word in inappropriate_words for word in words)

    def get_ml_response(self, user_input):
        """Get response using ML model"""
        try:
            # Preprocess input
            cleaned_input = self.preprocess_input(user_input)
            
            # Get prediction and confidence
            prediction = self.pipeline.predict([cleaned_input])[0]
            confidence = max(self.pipeline.predict_proba([cleaned_input])[0])
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"ML response error: {str(e)}")
            return None, 0.0

    def get_contextual_fallback_response(self, analysis):
        """Get context-aware fallback responses"""
        if analysis['is_question']:
            return "Could you please provide more details about your question? I want to give you the most accurate information about our services."
        
        if analysis['has_location']:
            return "I notice you're asking about a specific location. Could you specify what type of property information you're looking for in that area?"
        
        if analysis['has_price']:
            return "I see you're interested in pricing. Are you looking to buy, sell, or rent a property?"
        
        return "I'm not quite sure what you're looking for. Could you tell me more about your property needs?"

    def get_property_type_response(self, property_type):
        """Get property type specific responses"""
        responses = {
            'residential': "I can help you with residential properties. Are you looking to buy, sell, or rent?",
            'commercial': "For commercial properties, we have various options available. What kind of commercial space are you interested in?",
            'land': "I can assist you with land properties. Are you looking for developed plots or agricultural land?"
        }
        return responses.get(property_type, "What kind of property are you interested in?")

    def update_memory(self, user_id, interaction):
        """Update conversation memory"""
        self.conversation_memory[user_id].append(interaction)
        if len(self.conversation_memory[user_id]) > self.memory_size:
            self.conversation_memory[user_id].pop(0)

    def _get_conversation_context(self, user_id):
        """Get conversation context for user"""
        if user_id in self.conversation_memory:
            return {
                'interaction_count': len(self.conversation_memory[user_id]),
                'previous_sentiments': [i.get('analysis', {}).get('sentiment', 'neutral') 
                                      for i in self.conversation_memory[user_id]],
                'topics': self._analyze_frequent_topics(user_id)
            }
        return {}

    def _analyze_frequent_topics(self, user_id):
        """Analyze frequent topics in conversation"""
        topics = defaultdict(int)
        if user_id in self.conversation_memory:
            for interaction in self.conversation_memory[user_id]:
                input_text = interaction.get('input', '')
                for prop_type in self.property_types:
                    if re.search(self.property_types[prop_type], input_text, re.IGNORECASE):
                        topics[prop_type] += 1
        return dict(topics)

    def enhance_response(self, base_response, analysis, context):
        """Enhance response with context and analysis"""
        if not base_response:
            return self.get_contextual_fallback_response(analysis)

        response = base_response

        # Add urgency handling
        if analysis['sentiment'] == 'urgent':
            response = "I understand this is urgent. " + response

        # Add property type specific information
        if analysis['property_type'] != 'general':
            prop_type_info = {
                'residential': " We have extensive residential property listings you might be interested in.",
                'commercial': " Our commercial property section has various options for businesses.",
                'land': " We have many land parcels available for development."
            }
            response += prop_type_info.get(analysis['property_type'], '')

        return response

    def get_response(self, user_input, user_id='default'):
        """Main response generation method"""
        if not isinstance(user_input, str) or not user_input.strip():
            return "I couldn't understand that. How can I help you with Nextopson's services?"
        
        try:
            # Clean and analyze input
            cleaned_input = self.preprocess_input(user_input)
            analysis = self.analyze_input(cleaned_input)
            
            # Check for inappropriate language
            if self.contains_inappropriate_language(cleaned_input):
                return "Let's keep our conversation professional. How can I assist you with your property needs?"
            
            # Get conversation context
            context = self._get_conversation_context(user_id)
            
            # Try pattern matching first
            chat_response = self.chat.respond(cleaned_input)
            if chat_response:
                self.update_memory(user_id, {
                    'input': cleaned_input,
                    'response': chat_response,
                    'analysis': analysis,
                    'confidence': 0.5
                })
                return chat_response
            
            # Use ML model
            ml_response, confidence = self.get_ml_response(cleaned_input)
            if ml_response and confidence > 0.4:
                enhanced_response = self.enhance_response(ml_response, analysis, context)
                self.update_memory(user_id, {
                    'input': cleaned_input,
                    'response': enhanced_response,
                    'analysis': analysis,
                    'confidence': float(confidence)
                })
                return enhanced_response
            
            # If confidence is low but we have a property type
            if analysis['property_type'] != 'general':
                return self.get_property_type_response(analysis['property_type'])
                
            return self.get_contextual_fallback_response(analysis)
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I'm having trouble processing your question. Could you please rephrase it?"

def initialize_bot():
    """Initialize the bot"""
    try:
        bot = NextopsonSupportBot()
        logger.info("Bot initialized successfully")
        return bot
    except Exception as e:
        logger.error(f"Bot initialization error: {str(e)}")
        raise

# Initialize bot when module is loaded
bot = initialize_bot()