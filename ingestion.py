import re
import os
import tempfile
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WikipediaLoader
from langchain_core.documents import Document as LCDocument
from models import Document, DocumentChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1. Text Cleaning Utility
def clean_text(text: str) -> str:
    """Performs noise removal and whitespace normalization."""
    # Remove redundant whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters / artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()

# 2. Normalized Ingestion Pipeline
class DocumentIngestor:
    def process_files(self, uploaded_files):
        """Processes Streamlit uploaded files into a list of LangChain Document objects."""
        all_docs = []
        
        for uploaded_file in uploaded_files:
            # 1. Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # 2. Choose the correct loader based on file extension
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(tmp_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                # 3. Load and add to our list
                docs = loader.load()
                # Manually add metadata to help FAISS later
                for doc in docs:
                    doc.metadata["title"] = uploaded_file.name
                    doc.metadata["source_type"] = "local"
                
                all_docs.extend(docs)
                
            finally:
                # 4. Clean up the temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        # 5. Return the list of Document objects
        return all_docs
    

# 4. Chunking Strategy
def chunk_documents(documents: List[Document]) -> List[DocumentChunk]:
    """Splits Documents into smaller overlapping chunks with rich metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunk_objs = []
    for doc in documents:
        # split_text returns a list of strings
        texts = text_splitter.split_text(doc.content)
        
        for i, text in enumerate(texts):
            chunk_objs.append(DocumentChunk(
                parent_id=doc.source_id,
                source_type=doc.source_type,
                title=doc.title,
                content=text,
                metadata={
                    "chunk_index": i,
                    "title": doc.title,
                    "source_type": doc.source_type
                }
            ))
    return chunk_objs

# 5 & 6. FAISS Vector Store & Semantic Search
class VectorEngine:
    def __init__(self):
        # Initialize your embedding model here
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.save_path = "faiss_index"

    def index_documents(self, chunks):
        """Creates a FAISS index from document chunks and saves it locally."""
        if not chunks:
            return None
        
        # 1. Create the vector store from chunks
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # 2. Save the index to a local folder for persistence
        self.vector_store.save_local(self.save_path)
        print(f"Index successfully saved to {self.save_path}")
        
        return self.vector_store

    def load_faiss_index(self):
        """Loads index from disk with required security flag."""
        try:
            self.vector_store = FAISS.load_local(
                self.save_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return True
        except Exception:
            return False
        
    def semantic_search(self, query: str, k: int = 3):
        """
        Finds the most relevant document chunks for a given query.
        """
        # 1. Safety check: If the index isn't in memory, try loading it from disk
        if self.vector_store is None:
            success = self.load_faiss_index()
            if not success:
                # If still nothing, return an empty list so the app doesn't crash
                return []

        # 2. Use LangChain's similarity search
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

# Initialize Groq for the Generation part later
def get_groq_llm():
    return ChatGroq(
        model="llama3-8b-8192", # Or "llama3-70b-8192" for higher reasoning
        groq_api_key=os.getenv("GROQ_API_KEY")
    )