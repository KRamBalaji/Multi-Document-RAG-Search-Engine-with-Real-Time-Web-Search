# Multi-Document-RAG-Search-Engine-with-Real-Time-Web-Search

A sophisticated Retrieval-Augmented Generation (RAG) application that blends your private PDF data with real-time intelligence from the web.


## 🌟 Key Features
- **Multi-PDF Indexing**: Upload and search across multiple documents simultaneously.
- **Hybrid Intelligence**: Toggle between internal document search and real-time web search (Tavily).
- **Source Attribution**: Transparent answers that cite specific PDF chunks or web URLs.
- **Smart Routing**: Automatically classifies queries to determine if they need documents, web, or both.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.1)
- **Vector Store**: FAISS
- **Web Search**: Tavily API
- **Orchestration**: LangChain

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   🤖 SMART-RAG SYSTEM ARCHITECTURE               │
└──────────────────────────────────────────────────────────────────┘
         
    [ 📂 DATA INGESTION ]             [ 👤 USER INTERFACE ]
    ┌───────────────────┐             ┌───────────────────┐
    │  PDF Documents    │             │   Streamlit App   │
    └─────────┬─────────┘             └─────────┬─────────┘
              │                                 │ (1) User Input
              ▼                                 ▼
    ┌───────────────────┐             ┌───────────────────┐
    │ Chunking & Embeds │             │  AGENCY ROUTER    │
    └─────────┬─────────┘             └─────────┬─────────┘
              │                                 │ (2) Selective Routing
              ▼                                 │
    ┌───────────────────┐        ┌──────────────┴───────────────┐
    │   FAISS VECTOR    │<───────┤  [A] INTERNAL: PDF Docs      │
    │      STORAGE      │        ├──────────────────────────────┤
    └───────────────────┘        │  [B] EXTERNAL: Wikipedia     │
                                 ├──────────────────────────────┤
                                 │  [C] LIVE: Tavily Web Search │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼ (3) Evidence Gathering
    ┌───────────────────┐             ┌───────────────────┐
    │ FINAL GENERATION  │<────────────┤ CONTEXT ASSEMBLER │
    │ (Groq / Llama 3)  │             └───────────────────┘
    └─────────┬─────────┘
              │ (4) Synthesized Response
              ▼
    ┌───────────────────┐
    │ 💬 CHAT DISPLAY   │
    └───────────────────┘
```

**Architecture Overview**

* **Ingestion Layer**: Processes raw PDFs into searchable "chunks" using PyPDF and tiktoken-optimized splitting, then stores them in a local **FAISS** index.

* **Orchestration Layer**: Uses LangChain to manage the "brain" of the app. It decides whether to look in your files, search the web, or both.

* **Retrieval Layer**: Combines semantic search (finding meaning in your PDFs) with real-time web search via the Tavily API.

* **Generation Layer**: A high-speed Llama 3.1 model hosted on Groq synthesizes the gathered evidence into a single, cohesive response.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/KRamBalaji/Multi-Document-RAG-Search-Engine-with-Real-Time-Web-Search/
cd multi-doc-rag
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a .env file in the root directory and add your API keys:
```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 4. Run the application
```
streamlit run app.py
```

## 📝 How to Use
1. Upload: Use the sidebar to upload one or more PDF files.
2. Index: Click 'Index Documents' to process them into the vector store.
3. Toggle: Decide if you want "Real-Time Web Search" turned ON or OFF.
4. Chat: Ask questions! The system will synthesize an answer from your files and the internet.


**Live Link -** https://multi-document-rag-search-engine-with-real-time-web-search.streamlit.app/
