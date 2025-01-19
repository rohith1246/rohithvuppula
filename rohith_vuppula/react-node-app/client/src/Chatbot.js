import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { user: 'bot', text: 'Hello! How can I help you today?' }
  ]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Toggle chat window visibility
  const toggleChatbot = () => setIsOpen(!isOpen);

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle message sending
  const handleSendMessage = async () => {
    if (userInput.trim()) {
      const newMessage = { user: 'user', text: userInput.trim() };
      setMessages(prev => [...prev, newMessage]);
      setUserInput('');
      setIsLoading(true);

      try {
        const response = await fetch('http://localhost:5000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ user_input: userInput.trim() }), // Match the backend's expected parameter name
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const botReply = { 
          user: 'bot', 
          text: data.response || 'Sorry, I didn\'t understand that.' 
        };
        setMessages(prev => [...prev, botReply]);
      } catch (error) {
        console.error('Error:', error);
        const botReply = { 
          user: 'bot', 
          text: 'Sorry, there was an error processing your request.' 
        };
        setMessages(prev => [...prev, botReply]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="chatbot-container">
      <button 
        className="chatbot-toggle" 
        onClick={toggleChatbot}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? 'Close Chat' : 'Chat with us'}
      </button>

      {isOpen && (
        <div className="chatbot-popup">
          <div className="chatbot-header">
            <h3>Chatbot</h3>
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, index) => (
              <div 
                key={`${index}-${msg.user}`} 
                className={`message ${msg.user}`}
              >
                <p>{msg.text}</p>
              </div>
            ))}
            <div ref={messagesEndRef} />
            {isLoading && (
              <div className="message bot">
                <p>Typing...</p>
              </div>
            )}
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              disabled={isLoading}
            />
            <button 
              onClick={handleSendMessage} 
              disabled={isLoading || !userInput.trim()}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;