from typing import List
from langchain_groq import ChatGroq
from models import QueryRoute, WebSearchResult
from langchain_community.tools.tavily_search import TavilySearchResults
import os

def classify_query(query: str) -> str:
    llm = ChatGroq(model="llama-3-8b-8192", temperature=0)
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