import React from 'react';
import './App.css';
import Chatbot from './Chatbot';  // Import the Chatbot component

function App() {
  return (
    <div className="App">
      <h1>Welcome to Nextopson</h1>
      <p>Find your perfect property with no brokerage fees!</p>

      {/* Include the Chatbot component */}
      <Chatbot />
    </div>
  );
}

export default App;
