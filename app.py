"""
Local PDF RAG Chatbot - Main Application

A Streamlit application for querying PDF documents using RAG.
"""

import streamlit as st
import os
from pathlib import Path
from src.pdf_parser import PDFParser
from src.chunking import RecursiveChunker
from src.embeddings import EmbeddingModel
from src.retriever import RetrieverWithRerank
from src.generator import LLMGenerator


# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


def main():
    """Main application function."""
    
    # Title and description
    st.title("📄 Local PDF RAG Chatbot")
    st.markdown("""
    Upload PDF documents and ask questions about their content. 
    All processing happens locally using Llama 3.2 via Ollama.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=256,
            max_value=1024,
            value=512,
            step=128,
            help="Size of each text chunk in characters"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=200,
            value=100,
            step=25,
            help="Overlap between chunks in characters"
        )
        
        top_k = st.slider(
            "Top K Results",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of chunks to retrieve"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Sampling temperature for LLM"
        )
        
        st.divider()
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document to query"
    )
    
    if uploaded_file is not None and not st.session_state.documents_processed:
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file temporarily
                temp_path = Path("data/uploads") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Parse PDF
                parser = PDFParser()
                text = parser.extract_text(str(temp_path))
                
                # Chunk text
                chunker = RecursiveChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = chunker.split(text)
                
                # Generate embeddings
                embedder = EmbeddingModel()
                embeddings = embedder.encode(chunks, show_progress=True)
                
                # Store in session state
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.documents_processed = True
                
                st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    # Display processing status
    if st.session_state.documents_processed:
        st.info(f"📊 Document loaded: {len(st.session_state.chunks)} chunks available")
    
    # Chat interface
    st.divider()
    st.header("💬 Chat with your Document")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        if not st.session_state.documents_processed:
            st.error("Please upload a PDF document first!")
        else:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve relevant chunks
                        retriever = RetrieverWithRerank(
                            embeddings=st.session_state.embeddings,
                            chunks=st.session_state.chunks
                        )
                        relevant_chunks = retriever.search(prompt, top_k=top_k)
                        
                        # Generate response
                        generator = LLMGenerator(model='llama3.2')
                        response = generator.generate(
                            query=prompt,
                            context=relevant_chunks,
                            temperature=temperature
                        )
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add assistant message to history
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Show retrieved chunks in expander
                        with st.expander("🔍 View Retrieved Chunks"):
                            for i, chunk in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Chunk {i}** (Score: {chunk['rerank_score']:.3f})")
                                st.text(chunk['text'][:300] + "...")
                                st.divider()
                    
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.info("Make sure Ollama is running and llama3.2 model is installed.")
                        st.code("ollama pull llama3.2")


if __name__ == "__main__":
    main()
