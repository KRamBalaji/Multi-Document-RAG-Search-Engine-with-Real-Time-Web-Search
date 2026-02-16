from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingestion import VectorEngine
from models import QueryRoute, WebSearchResult
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

HYBRID_PROMPT_TEMPLATE = """
You are an expert research assistant. You have been provided with two types of context:
1. [INTERNAL DOCUMENTS]: Private data from the user's uploaded files.
2. [WEB SEARCH]: Real-time information from the internet.

### GOAL:
Provide a comprehensive, synthesized answer. If the sources disagree, prioritize the [INTERNAL DOCUMENTS] for company-specific facts, but mention the [WEB SEARCH] for broader market context.

### CONTEXT:
{context_text}

### USER QUESTION:
{query}

### INSTRUCTIONS:
- Do not repeat the context verbatim.
- If an answer requires both sources, blend them naturally.
- Cite sources using [Doc] for internal and [Web] for internet sources.
- If the context is insufficient, say so.

### FINAL ANSWER:
"""

def classify_query(query: str) -> str:
    llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", # Updated model name
    groq_api_key=os.getenv("GROQ_API_KEY")
    )
    structured_llm = llm.with_structured_output(QueryRoute)
    
    system_prompt = """You are an expert query router. 
    - Use 'internal' for questions about specific documents, notes, or uploaded PDF content.
    - Use 'web' for current events, real-time statistics, or news (post-2024).
    - Use 'hybrid' if the query requires comparing internal data with external trends."""
    
    result = structured_llm.invoke([
        ("system", system_prompt),
        ("human", query)
    ])
    return result.logic

vector_engine = VectorEngine()

def gather_context(query: str, web_enabled: bool):
    context = {"docs": [], "web": [], "route": "internal"}
    
    # 1. Classify the query normally
    route = classify_query(query)
    
    # 2. STRICT GATE: If toggle is OFF, force internal regardless of classification
    if not web_enabled:
        route = "internal"
    
    context["route"] = route

    # 3. Document Search (Runs for 'internal' or 'hybrid')
    if route in ["internal", "hybrid"]:
        context["docs"] = vector_engine.semantic_search(query, k=5)
        
    # 4. Web Search (ONLY runs if toggle is ON and route is 'web' or 'hybrid')
    if web_enabled and route in ["web", "hybrid"]:
        context["web"] = execute_web_search(query, k=3)
        
    return context

def execute_web_search(query: str, k: int = 3):
    search = TavilySearchResults(max_results=k, search_depth="basic") # Use basic for reliability
    return search.invoke(query)

def generate_hybrid_answer(query, context):
    # 1. Format Internal Docs with clear Source headers
    doc_segments = []
    for d in context.get("docs", []):
        # Extract filename from metadata we saved during ingestion
        source_name = d.metadata.get("title", "Unknown Document")
        doc_segments.append(f"--- SOURCE: {source_name} ---\n{d.page_content}")
    
    doc_text = "\n\n".join(doc_segments)
    
    # 2. Extract Web Results Safely
    web_list = []
    for w in context.get("web", []):
        # Try dictionary style first
        if isinstance(w, dict):
            content = w.get("content", "")
        # Then try object attribute style (Standard for newer LangChain)
        elif hasattr(w, "page_content"):
            content = w.page_content
        # Fallback if it's a raw string or something else
        else:
            content = str(w)
        
        if content:
            web_list.append(f"[Web] {content}")
            
    web_text = "\n".join(web_list)
    
    # 3. Combine with clear headers
    combined_context = f"INTERNAL DOCUMENTS:\n{doc_text}\n\nWEB DATA:\n{web_text}"
    
    # 4. Initialize LLM (Ensure you use a capable model like llama-3.1-8b-instant)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    # 5. Run the chain
    prompt = ChatPromptTemplate.from_template(HYBRID_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"query": query, "context_text": combined_context})

def generate_source_summaries(context: dict, n: int = 3):
    """Generates concise summaries for the top N unique documents found."""
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
    summaries = []
    
    # Extract unique documents from the top chunks
    seen_sources = set()
    top_docs = []
    for doc in context.get("docs", []):
        title = doc.metadata.get("title")
        if title not in seen_sources:
            top_docs.append(doc)
            seen_sources.add(title)
        if len(top_docs) >= n: break

    # Summarize each unique document
    for doc in top_docs:
        summary_prompt = f"Summarize the key points of this document snippet in 2 sentences:\n\n{doc.page_content}"
        summary = llm.invoke(summary_prompt).content
        summaries.append({
            "title": doc.metadata.get("title"),
            "summary": summary,
            "type": "Doc"
        })
        
    return summaries