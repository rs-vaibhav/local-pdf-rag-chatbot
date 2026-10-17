# Local PDF RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot for querying PDF documents using Llama 3.2 and Ollama.

## Overview

This project implements a complete RAG pipeline that allows you to:
- Upload and parse PDF documents
- Chunk text recursively for optimal retrieval
- Perform semantic search using cosine similarity
- Re-rank retrieved chunks using a cross-encoder
- Query documents using a locally running LLM (Llama 3.2)

## Features

- **100% Local**: No API calls to external services, all processing happens locally
- **PDF Parsing**: Extract text from PDF documents efficiently
- **Recursive Text Chunking**: Split documents into semantically meaningful chunks
- **Semantic Retrieval**: Find relevant chunks using vector embeddings and cosine similarity
- **Cross-Encoder Reranking**: Improve retrieval accuracy by reordering chunks
- **Streamlit UI**: User-friendly web interface for interacting with the chatbot

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Ollama**: Local LLM runtime
- **Llama 3.2**: Language model for generating responses
- **NumPy**: Numerical computations
- **Sentence Transformers**: For generating embeddings
- **Cross-Encoder**: For reranking retrieved chunks
- **PyPDF2 / pdfplumber**: PDF parsing

## Project Structure

```
local-pdf-rag-chatbot/
├── app.py                   # Main Streamlit application
├── src/                     # Source code
│   ├── __init__.py
│   ├── pdf_parser.py        # PDF parsing utilities
│   ├── chunking.py          # Text chunking strategies
│   ├── embeddings.py        # Embedding generation
│   ├── retriever.py         # Semantic retrieval with reranking
│   └── generator.py         # LLM response generation
├── data/                    # Data directory
│   ├── uploads/             # Uploaded PDFs
│   └── cache/               # Cached embeddings
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Installation

### Prerequisites

1. Install Ollama:
   - macOS/Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Windows: Download from [ollama.com](https://ollama.com)

2. Pull Llama 3.2 model:
   ```bash
   ollama pull llama3.2
   ```

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd local-pdf-rag-chatbot
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Terminal Mode (CLI)

Run directly in your terminal:
```bash
# Chat with multiple PDFs:
python cli.py --pdf doc1.pdf doc2.pdf

# Or run interactively (will prompt for files):
python cli.py
```

### Option 2: Web Interface (Streamlit)

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload one or more PDF documents using the multi-file uploader

4. Start chatting across all your documents!

### Programmatic Usage

```python
from src.pdf_parser import PDFParser
from src.chunking import RecursiveChunker
from src.embeddings import EmbeddingModel
from src.retriever import RetrieverWithRerank
from src.generator import LLMGenerator

# Parse PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

# Chunk text
chunker = RecursiveChunker(chunk_size=500, overlap=50)
chunks = chunker.split(text)

# Generate embeddings
embedder = EmbeddingModel()
embeddings = embedder.encode(chunks)

# Retrieve relevant chunks
retriever = RetrieverWithRerank(embeddings, chunks)
relevant_chunks = retriever.search("your query", top_k=5)

# Generate response
generator = LLMGenerator(model="llama3.2")
response = generator.generate(query="your query", context=relevant_chunks)
print(response)
```

## How It Works

### 1. PDF Parsing
The system extracts text from uploaded PDF documents while preserving structure and handling various PDF formats.

### 2. Recursive Text Chunking
Documents are split into overlapping chunks using recursive character splitting, ensuring semantic coherence.

### 3. Embedding Generation
Each chunk is converted to a dense vector representation using a pre-trained sentence transformer model.

### 4. Semantic Retrieval
When a query is received:
- The query is embedded using the same model
- Cosine similarity is computed between query and chunk embeddings
- Top-k most similar chunks are retrieved

### 5. Cross-Encoder Reranking
Retrieved chunks are reranked using a cross-encoder model for improved relevance scoring.

### 6. Response Generation
The top reranked chunks are combined with the query and sent to Llama 3.2 via Ollama for response generation.

## Configuration

You can customize the following parameters in the app:

- **Chunk Size**: 256-1024 tokens (default: 512)
- **Chunk Overlap**: 50-200 tokens (default: 100)
- **Top K Results**: 3-10 chunks (default: 5)
- **Temperature**: 0.0-1.0 (default: 0.7)

## Performance Optimization

- **Caching**: Embeddings are cached to avoid recomputation
- **Batch Processing**: Embeddings are generated in batches for efficiency
- **Quantization**: Uses quantized models for faster inference

## Limitations

- Performance depends on the quality of PDF text extraction
- Very large documents may require significant memory
- Accuracy depends on the Llama 3.2 model capabilities

## Future Improvements

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Multi-document conversations
- [ ] Conversation history and context management
- [ ] Advanced chunking strategies (semantic chunking)
- [ ] Support for multiple embedding models
- [ ] Docker containerization

## Troubleshooting

### Common Issues

1. **Ollama not running**: Make sure Ollama service is running
   ```bash
   ollama serve
   ```

2. **Model not found**: Pull the required model
   ```bash
   ollama pull llama3.2
   ```

3. **Memory issues**: Reduce chunk size or batch size in configuration

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Ollama](https://ollama.com) for local LLM runtime
- [Meta AI](https://ai.meta.com/) for Llama models
- [Hugging Face](https://huggingface.co) for transformer models
- [Streamlit](https://streamlit.io) for the web framework
