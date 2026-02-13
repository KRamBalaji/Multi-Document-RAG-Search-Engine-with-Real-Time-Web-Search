import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WikipediaLoader
from langchain_core.documents import Document as LCDocument
from models import Document

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