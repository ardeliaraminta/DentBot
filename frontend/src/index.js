import React from 'react';
import ReactDOM from 'react-dom';
import App from './App'; // Import your root component

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root') // Mount to the element with id="root"
);
