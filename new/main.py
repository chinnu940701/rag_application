from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from typing import Dict, List, Optional
import uuid
import os
os.environ ["OPENAI_API_KEY"]='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'
# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history and document stores
chat_histories: Dict[str, List[Dict[str, str]]] = {}
document_stores: Dict[str, FAISS] = {}

# Models for requests and responses
class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    doc_id: str  # Add doc_id here

class QueryResponse(BaseModel):
    result: str
    chat_id: str

class ChatHistoryResponse(BaseModel):
    chat_histories: Dict[str, List[Dict[str, str]]]

# Load and process PDF
def process_pdf(file_path: str) -> str:
    """Reads and extracts text from a PDF file."""
    pdf_reader = PdfReader(file_path)
    raw_text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Prepare embeddings and search index
def prepare_search_index(text: str) -> FAISS:
    """Splits text into chunks, generates embeddings, and creates a FAISS vector store."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# Initialize QA chain
def get_chain():
    """Returns a QA chain using the OpenAI LLM."""
    return load_qa_chain(OpenAI(), chain_type="stuff")

chain = get_chain()

# API Endpoints
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF uploads, processes them, and creates a vector store."""
    # Save the uploaded file locally
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Process PDF and prepare embeddings
        raw_text = process_pdf(file_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no readable text.")
        
        document_store = prepare_search_index(raw_text)

        # Generate a unique ID for the document store
        doc_id = str(uuid.uuid4())
        document_stores[doc_id] = document_store

        return {"message": "PDF uploaded and processed successfully", "doc_id": doc_id}
    finally:
        # Clean up the temporary file
        os.remove(file_path)


#@app.post("/query", response_model=QueryResponse)
#async def query(request: QueryRequest, doc_id: str):
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query = request.query
    doc_id = request.doc_id
    chat_id = request.chat_id or "default"

    """Handles queries against a specific document store."""
    query = request.query
    chat_id = request.chat_id or "default"

    # Check if the document store exists
    if doc_id not in document_stores:
        raise HTTPException(status_code=404, detail="Document store not found")

    # Perform similarity search
    document_store = document_stores[doc_id]
    docs = document_store.similarity_search(query)
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
    """Retrieves the chat history for all chats."""
    return ChatHistoryResponse(chat_histories=chat_histories)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
