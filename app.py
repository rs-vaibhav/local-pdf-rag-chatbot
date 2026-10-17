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
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
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
    Upload one or more PDF documents and ask questions across all of them. 
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
        
        if st.button("💬 Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()

        if st.button("🗑️ Reset All (Docs & Chat)"):
            st.session_state.conversation_history = []
            st.session_state.chunks = []
            st.session_state.embeddings = None
            st.session_state.documents_processed = False
            st.session_state.processed_files = []
            st.rerun()
    
    # File uploader - supports multiple files
    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to query"
    )
    
    current_filenames = [f.name for f in uploaded_files] if uploaded_files else []
    
    # Re-process if uploaded files changed
    if uploaded_files and (not st.session_state.documents_processed or current_filenames != st.session_state.processed_files):
        new_files = [fn for fn in current_filenames if fn not in st.session_state.processed_files]
        is_mid_chat_update = bool(st.session_state.processed_files)
        
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            try:
                parser = PDFParser()
                chunker = RecursiveChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                all_chunks = []
                all_chunk_texts = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file
                    temp_path = Path("data/uploads") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Parse PDF
                    text = parser.extract_text(str(temp_path))
                    
                    # Chunk text
                    file_chunks = chunker.split(text)
                    for chunk_text in file_chunks:
                        all_chunks.append({
                            'text': chunk_text,
                            'source': uploaded_file.name
                        })
                        all_chunk_texts.append(chunk_text)
                
                if all_chunk_texts:
                    # Generate embeddings
                    embedder = EmbeddingModel()
                    embeddings = embedder.encode(all_chunk_texts, show_progress=True)
                    
                    # Store in session state
                    st.session_state.chunks = all_chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.documents_processed = True
                    st.session_state.processed_files = current_filenames
                    
                    # Notify in chat if this was added mid-conversation
                    if is_mid_chat_update and new_files:
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": f"📑 **Added new document(s)**: {', '.join(new_files)}. You can now ask questions across all {len(current_filenames)} loaded documents!"
                        })
                    
                    st.success(f"✅ Processed {len(uploaded_files)} document(s)! Created {len(all_chunks)} chunks total.")
                else:
                    st.warning("No text could be extracted from the uploaded document(s).")
                
            except Exception as e:
                st.error(f"Error processing document(s): {str(e)}")
    
    # Display processing status
    if st.session_state.documents_processed and st.session_state.chunks:
        docs_list = ", ".join(st.session_state.processed_files)
        st.info(f"📊 Loaded {len(st.session_state.processed_files)} document(s): **{docs_list}** ({len(st.session_state.chunks)} total chunks available)")
    
    # Chat interface
    st.divider()
    st.header("💬 Chat with your Documents")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document(s)..."):
        if not st.session_state.documents_processed or not st.session_state.chunks:
            st.error("Please upload at least one PDF document first!")
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
                                source_tag = f" • Source: `{chunk['source']}`" if chunk.get('source') else ""
                                st.markdown(f"**Chunk {i}** (Score: {chunk['rerank_score']:.3f}{source_tag})")
                                st.text(chunk['text'][:300] + "...")
                                st.divider()
                    
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.info("Make sure Ollama is running and llama3.2 model is installed.")
                        st.code("ollama pull llama3.2")


if __name__ == "__main__":
    main()
