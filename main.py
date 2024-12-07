import streamlit as st
import numpy as np
from transformers import pipeline
import faiss
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
 
# Initialize Streamlit app
st.title("PDF Question Answering System with FAISS and LangChain")
 
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() 
    return text
 
# Embedder class using LangChain for embedding generation
class Embedder:
    def __init__(self):
        # Use LangChain's HuggingFace embeddings wrapper
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    def embed_text(self, texts, is_query=False):
        if is_query:
            # For single query embedding
            return self.embedding_model.embed_query(texts)
        else:
            # For embedding documents (multiple chunks of text)
            return self.embedding_model.embed_documents(texts)
 
embedder = Embedder()
 
# Create a FAISS index with the appropriate dimension
dimension = 384  # The dimension for 'all-MiniLM-L6-v2' embeddings
index = faiss.IndexFlatL2(dimension)
 
# Function to add text chunks to the FAISS index
def add_texts_to_index(texts):
    embeddings = embedder.embed_text(texts)  # Embed all texts at once
    index.add(np.array(embeddings))
 
# Function to find the most relevant chunks
def find_most_relevant_chunks(query_embedding, k=3):
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]
 
# Set up the question-answering chain using FAISS and Hugging Face LLM
def answer_question(question):
    # Embed the question using LangChain's HuggingFace model
    question_embedding = embedder.embed_text(question, is_query=True)
    
    # Find the most relevant chunk indices
    relevant_chunk_indices = find_most_relevant_chunks(question_embedding)
    
    # Get the relevant text chunks
    relevant_texts = " ".join([texts[idx] for idx in relevant_chunk_indices])
    
    # Initialize the Hugging Face QA pipeline
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
 
    # Use the QA model to generate an answer
    answer = qa_pipeline({'question': question, 'context': relevant_texts})
 
    return answer['answer']
 
# Streamlit file uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    # Split the extracted text into chunks (e.g., by paragraphs)
    text_splitter = CharacterTextSplitter(separator='\n\n', chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(extracted_text)
    
    # Add extracted text chunks to the FAISS index
    add_texts_to_index(texts)
    
    # Display a text input for asking questions
    question = st.text_input("Ask a question:")
    
    if question:
        answer = answer_question(question)
        st.write("Answer:", answer)
