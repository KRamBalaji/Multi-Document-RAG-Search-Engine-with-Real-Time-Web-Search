import re
import os
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
    def __init__(self):
        self.documents: List[Document] = []

    def load_pdfs(self, file_paths: List[str]):
        for path in file_paths:
            loader = PyPDFLoader(path)
            raw_docs = loader.load()
            # Combine pages into one Document object for consistency
            full_content = " ".join([d.page_content for d in raw_docs])
            self.documents.append(Document(
                source_type="pdf",
                title=path.split("/")[-1],
                content=clean_text(full_content),
                metadata={"file_path": path}
            ))

    def load_wikipedia(self, queries: List[str]):
        for query in queries:
            loader = WikipediaLoader(query=query, load_max_docs=1)
            raw_docs = loader.load()
            for d in raw_docs:
                self.documents.append(Document(
                    source_type="wikipedia",
                    title=d.metadata.get("title", query),
                    content=clean_text(d.page_content),
                    metadata={"source_url": d.metadata.get("source")}
                ))

    def load_text_files(self, file_paths: List[str]):
        for path in file_paths:
            loader = TextLoader(path)
            d = loader.load()[0]
            self.documents.append(Document(
                source_type="text",
                title=path.split("/")[-1],
                content=clean_text(d.page_content),
                metadata={"file_path": path}
            ))

    def get_all_documents(self) -> List[Document]:
        return self.documents
    

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
    def __init__(self, index_path="faiss_index"):
        # We use a standard open-source embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_path = index_path
        self.vector_store = None

    def index_documents(self, chunks: List[DocumentChunk]):
        """Creates and persists a FAISS index from document chunks."""
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        self.vector_store.save_local(self.index_path)
        print(f"Index saved to {self.index_path}")

    def load_faiss_index(self):
        """Loads the local FAISS index."""
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print("No index found. Please run index_documents first.")

    def semantic_search(self, query: str, k: int = 4):
        """Finds the most relevant document chunks for a query."""
        if not self.vector_store:
            self.load_faiss_index()
            
        # Returns a list of LangChain Document objects
        results = self.vector_store.similarity_search(query, k=k)
        return results

# Initialize Groq for the Generation part later
def get_groq_llm():
    return ChatGroq(
        model="llama3-8b-8192", # Or "llama3-70b-8192" for higher reasoning
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )