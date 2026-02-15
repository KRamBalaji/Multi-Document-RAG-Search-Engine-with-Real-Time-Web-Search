from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingestion import VectorEngine
from models import QueryRoute, WebSearchResult
from langchain_community.tools.tavily_search import TavilySearchResults
import os

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

def gather_context(query: str):
    route = classify_query(query)
    context = {"docs": [], "web": [], "route": route}
    
    if route in ["internal", "hybrid"]:
        # Retrieve from our VectorEngine(FAISS)
        context["docs"] = VectorEngine.semantic_search(query, k=3)
        
    if route in ["web", "hybrid"]:
        # Retrieve from Tavily
        context["web"] = execute_web_search(query, k=3)
        
    return context

def execute_web_search(query: str, k: int = 3) -> List[WebSearchResult]:
    """Executes a real-time search and returns standardized web results."""
    search_tool = TavilySearchResults(
        max_results=k,
        tavily_api_key=os.environ.get("TAVILY_API_KEY")
    )
    
    raw_results = search_tool.invoke({"query": query})
    web_docs = []
    
    for res in raw_results:
        # res typically contains 'content' and 'url'
        web_docs.append(WebSearchResult(
            title=res.get("title", "Web Result"),
            url=res.get("url"),
            content=res.get("content"),
            score=res.get("score", 0.0)
        ))
    
    return web_docs

def generate_hybrid_answer(query: str, context: dict):
    """
    Combines FAISS chunks and Tavily snippets into a cited answer.
    """
    llm = ChatGroq(model="llama3-7b-8192", temperature=0)
    
    # 1. Format Document Chunks
    doc_context = ""
    for i, doc in enumerate(context.get("docs", [])):
        source_title = doc.metadata.get("title", "Unknown Doc")
        doc_context += f"\n[Doc] {source_title} (Chunk {i}): {doc.page_content}\n"

    # 2. Format Web Results
    web_context = ""
    for res in context.get("web", []):
        web_context += f"\n[Web] Tavily: {res.title} (URL: {res.url}): {res.content}\n"

    # 3. Construct System Prompt
    system_msg = """You are a grounded research assistant. 
    Use the provided context to answer the user's question. 
    
    RULES:
    1. Only use the provided context. If unsure, say you don't know.
    2. Cite every claim using the exact format: [Doc] Title – Chunk# or [Web] Tavily: Title.
    3. Keep the tone professional and objective.
    
    CONTEXT:
    {doc_context}
    {web_context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "doc_context": doc_context,
        "web_context": web_context,
        "query": query
    })
    
    return response.content

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