from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from uuid import uuid4

class Document(BaseModel):
    """Represents a full raw document ingested into the system."""
    source_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: str  # 'pdf', 'wikipedia', 'text', 'markdown'
    title: str
    content: str
    metadata: Dict[str, Any] = {}

class DocumentChunk(BaseModel):
    """Represents a granular piece of text stored in the FAISS vector index."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str  # Links back to Document.source_id
    source_type: str
    title: str
    content: str
    metadata: Dict[str, Any]  # Includes chunk_index, page_number, etc.

class WebSearchResult(BaseModel):
    """Standardized format for real-time results from Tavily/Web."""
    source_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: str = "web"
    title: str
    url: HttpUrl
    content: str  # The snippet or full page text
    score: float  # Search relevance score

class AnswerSource(BaseModel):
    """The model used to display citations in the final UI."""
    title: str
    source_type: str
    url: Optional[str] = None
    snippet: str
    relevance_label: str  # E.g., 'Primary Evidence', 'Supporting Context'