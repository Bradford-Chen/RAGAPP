# Project Description
This project is a conversational application that utilizes various language models and embedding techniques to implement Q&A and dialogue functionalities. Users can choose different conversation modes, including normal mode, retrieval Q&A mode without history, and retrieval Q&A mode with history.  
# Main Packages and Versions
`streamlit`: Framework for building web applications
`langchain`: Library for building language model chains
`chromadb`: Library for vector databases
`zhipuai_embedding`: ZhipuAI embedding model
`.env`: Library for loading environment variables
# Environment Configuration
- **Python Version**: 3.8 and above  
- Install Dependencies:  
```
pip install streamlit langchain chromadb zhipuai_embedding python-dotenv
```
- **Environment Variables Configuration**: Create a .env file in the project root directory and add the following content:  
# Running the Project
- Start the Application:  
`streamlit run streamlit_app.py`
- **Access the Application**: Open `http://localhost:8501` in your browser to access the application interface.
