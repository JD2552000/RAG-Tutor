import streamlit as st
from src.ingestion.processor import DataIngestor
from src.database.vector_store import VectorDatabase
from src.engine import TutorEngine
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Enterprise AI Tutor", layout="wide")
st.title("🎓 Enterprise AI Tutor (Gemini Pro + M3)")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingestor" not in st.session_state:
    st.session_state.ingestor = DataIngestor()
if "db" not in st.session_state:
    st.session_state.db = VectorDatabase()
if "engine" not in st.session_state:
    st.session_state.engine = TutorEngine()

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload Study Materials", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.spinner("Analyzing and indexing documents..."):
                # Save uploaded files to data/docs temporarily
                for uploaded_file in uploaded_files:
                    with open(os.path.join("data/docs", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Ingest and Store
                chunks = st.session_state.ingestor.load_and_split("data/docs")
                st.session_state.db.add_documents(chunks)
                st.success("Indexing Complete!")
        else:
            st.error("Please upload files first.")

# --- Main Chat Interface ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your tutor anything..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.engine.ask(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})