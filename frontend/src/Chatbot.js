import React, { useState, useEffect } from 'react';
import { askQuestion } from './api';
import './style.css';

import botProfileImage from './facultyavatar.jpg';
import userImage from './useravatar.jpg';

const Chatbot = () => {
  const [userInput, setUserInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [isWaitingForConfirmation, setIsWaitingForConfirmation] = useState(false);
  const [suggestedPattern, setSuggestedPattern] = useState('');

  useEffect(() => {
    // Initialize chat with a greeting message from the bot
    setChatMessages([
      {
        text: "Hello! How can I assist you today?",
        isUser: false
      }
    ]);
  }, []);

  const handleUserInput = (e) => {
    setUserInput(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Trim the user input to remove leading and trailing whitespace
    const trimmedInput = userInput.trim();

    // Check if the trimmed input is empty
    if (!trimmedInput) {
      console.log('Please enter a message before sending.');
      return;
    }

    try {
      const response = await askQuestion(trimmedInput);

      const updatedChat = [
        ...chatMessages,
        { text: trimmedInput, isUser: true },
        { text: response, isUser: false }
      ];
      setChatMessages(updatedChat);

      if (response.includes("Did you mean:")) {
        const suggested = response.split("Did you mean: '")[1].split("'?")[0];
        setIsWaitingForConfirmation(true);
        setSuggestedPattern(suggested);
      } else {
        setIsWaitingForConfirmation(false);
        setSuggestedPattern('');
      }

      setUserInput('');

    } catch (error) {
      console.error('Error:', error);
      const errorMessage = 'An error occurred while processing your request.';
      const errorMessages = [
        ...chatMessages,
        { text: trimmedInput, isUser: true },
        { text: errorMessage, isUser: false }
      ];
      setChatMessages(errorMessages);
    }
  };

  const handleUserResponse = async (response) => {
    setIsWaitingForConfirmation(false);
    const updatedChat = [
      ...chatMessages,
      { text: response, isUser: true }
    ];
    setChatMessages(updatedChat);

    try {
      const botResponse = await askQuestion(userInput, response.toLowerCase(), suggestedPattern);
      const finalChat = [
        ...updatedChat,
        { text: botResponse, isUser: false }
      ];
      setChatMessages(finalChat);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = 'An error occurred while processing your request.';
      const errorMessages = [
        ...updatedChat,
        { text: errorMessage, isUser: false }
      ];
      setChatMessages(errorMessages);
    }
  };

  const botName = "Faculty";
  const userName = "You";

  return (
    <div className="chat-container">
      <div className="chat-box">
        {chatMessages.map((message, index) => (
          <div key={index} className={`message-container ${message.isUser ? 'user' : 'bot'}`}>
            {!message.isUser ? (
              <div className="bot-info">
                <img src={botProfileImage} alt={botName} className="bot-profile-image" />
                <div className="bot-name">{botName}</div>
              </div>
            ) : (
              <div className="user-info">
                <div className="user-name">{userName}</div>
                <img src={userImage} alt={userName} className="user-profile-image" />
              </div>
            )}
            <div className={`message ${message.isUser ? 'user-message' : 'bot-message'}`}>
              {message.text}
            </div>
          </div>
        ))}
        {/* Render confirmation buttons below the last bot message if waiting for confirmation */}
        {isWaitingForConfirmation && (
          <div className="confirmation-buttons">
            <button className="yes-button" onClick={() => handleUserResponse('yes')}>Yes</button>
            <button className="no-button" onClick={() => handleUserResponse('no')}>No</button>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="user-input-form">
        <input
          type="text"
          value={userInput}
          onChange={handleUserInput}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default Chatbot;
