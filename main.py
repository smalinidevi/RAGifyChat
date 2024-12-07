from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO 
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings 
import faiss
from langchain.docstore import InMemoryDocstore

import os
from langchain.chains import ConversationalRetrievalChain  
from langchain.memory import ConversationBufferMemory  
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import sys


os.environ['OPENAI_API_KEY'] = #openai_api_key
 
# Initialize memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  
app = FastAPI()
 
# Initialize FAISS vector store
def initialize_faiss_vector_store():
    # Step 1: Create embeddings instance
    embeddings = OpenAIEmbeddings(api_key=#openai_api_key)
 
    # Step 2: Define the dimension of your embeddings
    embedding_dimension = 1536  # Example dimension for OpenAI's models, adjust according to your model
 
    # Step 3: Initialize a FAISS index with L2 distance
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
 
    # Step 4: Create an InMemoryDocstore instance
    docstore = InMemoryDocstore({})  # Empty docstore initially
 
    # Step 5: Initialize the FAISS vector store using correct parameters
    vector_store = FAISS(embedding_function=embeddings.embed_query,
                         index=faiss_index,
                         docstore=docstore,
                         index_to_docstore_id={})
 
    return vector_store
 
# Initialize the vector store
vector_store = initialize_faiss_vector_store()

def getChatLLMChain(k):
    llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=, temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    
    system_template = """
    Use the following pieces of context to answer the user's question. If you don't find the answer in the provided context, just respond "I don't have the answer. Connecting you to a support agent." in plain text
    Context: {context}
    """
    
    user_template = "Question: {question}"
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]
    
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type='stuff',
        combine_docs_chain_kwargs={'prompt': qa_prompt},
        verbose=False
    )
    return chain
 
def ask_and_get_answer(vector_store, q, chain, k=3):
    answer = chain.invoke({'question': q})
    return answer['answer']
 
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or Excel file.")
 
    try:
        if file.content_type == "text/csv":
            df = pd.read_csv(BytesIO(await file.read()))
        else:
            df = pd.read_excel(BytesIO(await file.read()))
 
        return JSONResponse(content={"acknowledgment": "File received and processed successfully."}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")
  
@app.post("/chat/")
async def chat(prompt: str):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    usermsg = prompt
    print(usermsg)
    
    k = 3
    chain = getChatLLMChain(k)
    answer = ask_and_get_answer(vector_store, prompt, chain, k)
    
    translated_answer = answer
    response = translated_answer
    return {"response": response}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
