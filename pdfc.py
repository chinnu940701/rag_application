from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from typing import Dict, List, Optional
import uuid
from dotenv import load_dotenv, find_dotenv

#Load environment variables
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#print(OPENAI_API_KEY)
os.environ ["OPENAI_API_KEY"]='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'
app = FastAPI()

# In-memory chat history
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# Models for requests and responses
class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None

class QueryResponse(BaseModel):
    result: str
    chat_id: str

class ChatHistoryResponse(BaseModel):
    chat_histories: Dict[str, List[Dict[str, str]]]

# Load and process PDF
def process_pdf(file_path: str):
    pdf_reader = PdfReader(file_path)
    raw_text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Prepare embeddings and search index
def prepare_search_index(text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# Initialize chain
def get_chain():
    return load_qa_chain(OpenAI(), chain_type="stuff")

# Load PDF and create the search index
raw_text = process_pdf("INDIAN_ECONOMY.pdf")
document_search = prepare_search_index(raw_text)
chain = get_chain()

# API Endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query = request.query
    chat_id = request.chat_id or "default"

    # Perform similarity search
    docs = document_search.similarity_search(query)
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Get the result from the chain
    result = chain.run(input_documents=docs, question=query)

    # Update chat history
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_histories[chat_id].append({"query": query, "response": result})

    return QueryResponse(result=result, chat_id=chat_id)

@app.get("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history():
    return ChatHistoryResponse(chat_histories=chat_histories)
# Endpoint to start a new chat
# @app.post("/new_chat")
# async def new_chat():
#     chat_id = str(uuid.uuid4())
#     chat_histories[chat_id] = []
#     return {"chat_id": chat_id}



# To run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
