# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime

# Import the NextopsonSupportBot class
from nextopson_bot import NextopsonSupportBot, initialize_bot

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app and chatbot
app = Flask(__name__)
CORS(app)

# Initialize the chatbot
try:
    chatbot = initialize_bot()
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    raise

# Request logging middleware
@app.before_request
def log_request_info():
    logger.info('Headers: %s', request.headers)
    logger.info('Body: %s', request.get_data())

# Error handler for all exceptions
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"An error occurred: {str(error)}", exc_info=True)
    return jsonify({
        "error": "An internal server error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests from users
    
    Expected JSON payload:
    {
        "user_input": "string",
        "user_id": "string" (optional)
    }
    """
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            logger.warning("No data provided in request")
            return jsonify({
                "error": "No data provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Extract and validate user input
        user_input = data.get('user_input', '')
        if not user_input or not isinstance(user_input, str):
            logger.warning(f"Invalid user input: {user_input}")
            return jsonify({
                "error": "Invalid or missing user input",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Extract optional user ID
        user_id = data.get('user_id', 'default')
        
        # Get response from chatbot
        logger.info(f"Processing chat request - User ID: {user_id}, Input: {user_input}")
        response = chatbot.get_response(user_input, user_id)
        
        # Return response
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An error occurred processing your request",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify API is running"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def api_docs():
    """Return basic API documentation"""
    return jsonify({
        "name": "Nextopson Chatbot API",
        "version": "1.0",
        "endpoints": {
            "/chat": {
                "method": "POST",
                "description": "Send a message to the chatbot",
                "payload": {
                    "user_input": "string (required)",
                    "user_id": "string (optional)"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Check API health status"
            }
        },
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Nextopson Chatbot API")
    app.run(debug=True, host='0.0.0.0', port=5000)