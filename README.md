# RAGifyChat
This repository hosts an AI-powered chatbot that integrates Retrieval-Augmented Generation (RAG) and FAISS to provide efficient document retrieval and conversational abilities. Designed to handle user queries intelligently, the application processes uploaded CSV or Excel files and retrieves relevant information to generate meaningful responses.
 
Built with FastAPI and leveraging OpenAI's GPT models, the chatbot supports real-time interactions by combining document retrieval with conversational memory. The use of FAISS ensures high-speed, scalable vector searches for document context, making it ideal for knowledge-driven applications.

<h2>Features</h2>


<h3>Conversational AI</h3>

    Supports human-like conversational abilities using OpenAI GPT models.

<h3>Document Retrieval</h3>

    Upload and query documents (CSV or Excel) seamlessly.

    Uses FAISS for efficient vector-based similarity search.

<h3>Memory-Enhanced Conversations</h3>

    Retains conversation history using ConversationBufferMemory.

<h3>Embeddings</h3>

    Leverages OpenAI embeddings for high-dimensional vector storage and retrieval.

<h3>Flexible Input</h3>

    Upload CSV or Excel files for real-time information retrieval.

<h3>API-Based Architecture</h3>

    Built with FastAPI for robust, scalable backend support.

<h2>Installation</h2>


<h3>Prerequisites</h3>

    Python 3.8+

    OpenAI API Key

<h2>Steps</h2>

<h3>Clone this repository</h3>

    git clone https://github.com/smalinidevi/RAGifyChat.git  

    cd repo-name  

<h3>Install the dependencies</h3>

    pip install -r requirements.txt  

<h3>Set up the environment variables</h3>

    export OPENAI_API_KEY=your_openai_api_key  

<h3>Run the FastAPI application</h3>

     uvicorn main:app --host 0.0.0.0 --port 8000 

<h2>Endpoints</h2>

<h3>File Upload</h3>

    Endpoint: /upload/

    Method: POST

    Description: Uploads and processes a CSV or Excel file for use in the chatbot's document retrieval system.

    Payload: File - CSV or Excel

<h3>Chat</h3>

    Endpoint: /chat/

    Method: POST

    Description: Submits a user prompt and returns an AI-generated response using the document retrieval system.

    Payload: prompt - The user query or input.

<h2>How It Works</h2>

<h3>File Upload</h3>

    Upload CSV or Excel files via the /upload/ endpoint.

    Files are processed and indexed into the FAISS vector store.

<h3>Query Processing</h3>

    A user prompt is sent to the /chat/ endpoint.

    The chatbot retrieves relevant context from the FAISS vector store and generates a response using OpenAI GPT models.

<h3>Response Generation</h3>

    The chatbot combines retrieved context with the user query for accurate, context-aware responses.

<h2>Tech Stack</h2>


    Backend: FastAPI

    Embeddings: OpenAI GPT & FAISS

    Data Processing: Pandas

    Conversation Memory: LangChain

    Deployment: Uvicorn
