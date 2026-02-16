import streamlit as st
import os
from ingestion import DocumentIngestor, VectorEngine
from logic import gather_context, generate_hybrid_answer

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

# --- Page Configuration ---
st.set_page_config(page_title="Hybrid RAG Search", page_icon="🔍", layout="wide")

# Initialize session state for message history and document indexing
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingestor" not in st.session_state:
    st.session_state.ingestor = DocumentIngestor()
if "vector_engine" not in st.session_state:
    engine = VectorEngine()
    try:
        engine.load_faiss_index() # Attempt to load existing index
        st.session_state.vector_engine = engine
    except Exception:
        st.session_state.vector_engine = engine
        st.sidebar.warning("No existing index found. Please upload documents.")

# --- Sidebar: Document Management ---
if "indexed_filenames" not in st.session_state:
    st.session_state.indexed_filenames = []

with st.sidebar:
    st.title("📂 Knowledge Base")
    
    # Upload local documents
    uploaded_files = st.file_uploader(
        "Upload PDFs or Text Files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Index Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                # Your existing indexing logic
                chunks = st.session_state.ingestor.process_files(uploaded_files)
                st.session_state.vector_engine.index_documents(chunks)
                
                # NEW: Save the names of the files we just indexed
                new_names = [f.name for f in uploaded_files]
                st.session_state.indexed_filenames = list(set(st.session_state.indexed_filenames + new_names))
                
                st.success("✅ Index Updated!")
        else:
            st.error("Please upload files first.")

    # NEW: Display the list of currently indexed files
    if st.session_state.indexed_filenames:
        st.markdown("---")
        st.subheader("📚 Indexed Documents")
        for i, name in enumerate(st.session_state.indexed_filenames):
            st.caption(f"{i+1}. {name}")
    
    st.divider()
    
    # Toggle Web Search
    st.toggle("Enable Web Search", key="web_search_enabled")

# --- Main Interface ---
st.title("🤖 Hybrid RAG Search Engine")
st.markdown("Query your documents and the web simultaneously.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me something..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process Response
    with st.chat_message("assistant"):
        is_web_on = st.session_state.web_search_enabled
    
        with st.status("Searching sources...", expanded=True) as status:
            # Pass the boolean directly into your logic
            context = gather_context(prompt, web_enabled=is_web_on)
            status.update(label="Generating grounded answer...", state="running")
            
            # Generate final response
            answer = generate_hybrid_answer(prompt, context)
            status.update(label="Analysis complete!", state="complete")

            # Visual Indicators based on Route
            route_icons = {"internal": "📄 Document-based", "web": "🌐 Web-based", "hybrid": "🔀 Hybrid"}
            st.caption(f"Source Type: {route_icons.get(context['route'])}")
            
            # Display the Answer and Evidence Tabs
            tab_ans, tab_doc, tab_web = st.tabs(["💡 Answer", "📄 Doc Evidence", "🌐 Web Evidence"])
            
            with tab_ans:
                st.markdown(answer)
            
            with tab_doc:
                if context.get("docs"):
                    for d in context["docs"]:
                        st.info(f"**Source:** {d.metadata.get('title', 'Unknown')}\n\n{d.page_content}")
                else:
                    st.write("No document context found.")

            with tab_web:
                if not st.session_state.web_search_enabled:
                    st.warning("🚫 Web search is currently turned OFF in the sidebar.")
                elif context.get("web"):
                    for w in context["web"]:
                        # (Your safe dictionary-access code from before)
                        title = w.get('title', 'Web Result')
                        st.warning(f"**Source:** [{title}]({w.get('url')})\n\n{w.get('content')}")
                else:
                    st.write("No relevant web results found for this query.")

            # Save assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})