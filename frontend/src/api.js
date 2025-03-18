// import axios from 'axios';

// const baseURL = 'http://localhost:5000'; // Update with your Flask server URL
// const api = axios.create({
//   baseURL,
//   headers: {
//     'Content-Type': 'application/json',
//   },
// });

// export const askQuestion = async (userInput) => {
//   try {
//     const response = await api.post('/ask', { user_input: userInput });
//     return response.data.bot_response;
//   } catch (error) {
//     console.error('Error fetching bot response:', error);
//     return 'An error occurred while processing your request.';
//   }
// };

import axios from 'axios';

const baseURL = 'http://localhost:5000'; // Update with your Flask server URL
const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const askQuestion = async (userInput, userResponse = '', suggestedPattern = '') => {
  try {
    const requestBody = {
      user_input: userInput,
      user_response: userResponse,
      suggested_pattern: suggestedPattern
    };

    const response = await api.post('/ask', requestBody);
    return response.data.bot_response;
  } catch (error) {
    console.error('Error fetching bot response:', error);
    return 'An error occurred while processing your request.';
  }
};
