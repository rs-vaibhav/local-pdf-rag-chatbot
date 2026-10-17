"""
Local PDF RAG Chatbot - Terminal CLI Interface

Run and interact with the local PDF RAG chatbot directly in your terminal.
Supports adding new PDFs at startup or dynamically mid-chat using `/add <path>`.
"""

import sys
import os
import argparse
import shlex
import numpy as np
from pathlib import Path
from src.pdf_parser import PDFParser
from src.chunking import RecursiveChunker
from src.embeddings import EmbeddingModel
from src.retriever import RetrieverWithRerank
from src.generator import LLMGenerator


def parse_and_chunk_pdf(path: str, parser: PDFParser, chunker: RecursiveChunker):
    """Parse a single PDF and return a list of chunk dicts."""
    filename = Path(path).name
    text = parser.extract_text(path)
    if not text.strip():
        return [], 0
    raw_chunks = chunker.split(text)
    chunks = [{'text': c, 'source': filename} for c in raw_chunks]
    return chunks, len(text)


def print_help():
    print("""
Available in-chat commands:
  /add <path...>   Add one or more PDFs mid-chat (e.g. /add report.pdf ~/docs/memo.pdf)
  /docs or /files  List all currently indexed documents and their chunk counts
  /help            Show this help message
  exit or quit     Exit the chatbot
""")


def main():
    parser = argparse.ArgumentParser(description="Terminal interface for Local PDF RAG Chatbot")
    parser.add_argument("--pdf", nargs="+", help="Path(s) to one or more PDF documents to index")
    parser.add_argument("--model", type=str, default="llama3.2", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters (default: 100)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")

    args = parser.parse_args()

    print("=" * 65)
    print(" 📄 Local PDF RAG Chatbot - Terminal Mode")
    print("=" * 65)

    pdf_paths = []
    if args.pdf:
        for p in args.pdf:
            expanded = os.path.expanduser(p)
            if os.path.isfile(expanded):
                pdf_paths.append(expanded)
            else:
                print(f"Warning: File not found at '{p}', skipping.")
        if not pdf_paths:
            print("Error: None of the specified PDF files exist.")
            sys.exit(1)
    else:
        print("\nEnter initial PDF path(s). You can enter multiple paths separated by spaces,")
        print("or enter one now and add more later anytime using '/add <path>' mid-chat:")
        while True:
            try:
                line = input("PDF path: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                sys.exit(0)
            if not line:
                if pdf_paths:
                    break
                print("Please enter at least one PDF file path.")
                continue

            tokens = [t.strip() for t in line.replace(",", " ").split() if t.strip()]
            for token in tokens:
                expanded = os.path.expanduser(token)
                if os.path.isfile(expanded):
                    if expanded not in pdf_paths:
                        pdf_paths.append(expanded)
                        print(f"  + Added: {Path(expanded).name}")
                else:
                    print(f"  - File not found: '{token}'")

            if pdf_paths:
                break

    print(f"\n[1/4] 📖 Parsing {len(pdf_paths)} PDF document(s)...")
    pdf_parser = PDFParser()
    chunker = RecursiveChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    all_chunks = []
    all_chunk_texts = []
    loaded_docs = {}

    for path in pdf_paths:
        filename = Path(path).name
        print(f"  - Reading '{filename}'...")
        try:
            chunks, chars = parse_and_chunk_pdf(path, pdf_parser, chunker)
            if not chunks:
                print(f"    Warning: No text extracted from '{filename}'.")
                continue
            print(f"    Extracted {chars} chars -> {len(chunks)} chunks.")
            loaded_docs[filename] = len(chunks)
            for c in chunks:
                all_chunks.append(c)
                all_chunk_texts.append(c['text'])
        except Exception as e:
            print(f"    Error reading '{filename}': {e}")

    if not all_chunks:
        print("\n❌ Error: No text chunks could be extracted from any PDF.")
        sys.exit(1)

    print(f"\n[2/4] ✂️  Total chunks prepared: {len(all_chunks)} across {len(loaded_docs)} document(s).")

    print("[3/4] 🧠 Generating embeddings...")
    embedder = EmbeddingModel()
    embeddings = embedder.encode(all_chunk_texts, show_progress=True)

    print(f"[4/4] 🤖 Initializing Retriever & Ollama LLM ({args.model})...")
    retriever = RetrieverWithRerank(embeddings=embeddings, chunks=all_chunks)
    generator = LLMGenerator(model=args.model)

    print("\n" + "=" * 65)
    print(f"✅ Ready! Querying across: {', '.join(loaded_docs.keys())}")
    print("💡 Tip: You can add more PDFs anytime by typing: /add <path/to/doc.pdf>")
    print("Type '/help' for options, or 'exit' / 'q' to stop.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("\n💬 Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # In-chat command: help
        if query.lower() == "/help":
            print_help()
            continue

        # In-chat command: list docs
        if query.lower() in ("/docs", "/files", "/list"):
            print("\n📚 Currently Indexed Documents:")
            for doc, count in loaded_docs.items():
                print(f"  • {doc} ({count} chunks)")
            print(f"Total: {len(all_chunks)} chunks across {len(loaded_docs)} document(s).")
            continue

        # In-chat command: add new PDF mid-chat
        if query.startswith("/add ") or query.startswith("/load "):
            raw_paths = query.split(maxsplit=1)[1].strip()
            try:
                targets = shlex.split(raw_paths)
            except Exception:
                targets = [p.strip() for p in raw_paths.split() if p.strip()]

            new_paths = []
            for t in targets:
                expanded = os.path.expanduser(t)
                if os.path.isfile(expanded):
                    new_paths.append(expanded)
                else:
                    print(f"❌ File not found: '{t}'")

            if not new_paths:
                print("No valid files to add.")
                continue

            added_count = 0
            new_chunk_texts = []
            new_chunks = []

            for p in new_paths:
                fn = Path(p).name
                print(f"📖 Parsing and chunking '{fn}'...")
                try:
                    chunks, chars = parse_and_chunk_pdf(p, pdf_parser, chunker)
                    if not chunks:
                        print(f"  Warning: No text found in '{fn}'.")
                        continue
                    loaded_docs[fn] = loaded_docs.get(fn, 0) + len(chunks)
                    for c in chunks:
                        new_chunks.append(c)
                        new_chunk_texts.append(c['text'])
                    added_count += 1
                    print(f"  + Added '{fn}' ({len(chunks)} chunks, {chars} characters)")
                except Exception as e:
                    print(f"  Error processing '{fn}': {e}")

            if new_chunks:
                print("🧠 Generating embeddings for new document(s)...")
                new_embeddings = embedder.encode(new_chunk_texts, show_progress=True)
                all_chunks.extend(new_chunks)
                all_chunk_texts.extend(new_chunk_texts)
                embeddings = np.vstack([embeddings, new_embeddings])
                # Re-initialize retriever with updated embeddings and chunks
                retriever = RetrieverWithRerank(embeddings=embeddings, chunks=all_chunks)
                print(f"✅ Successfully added! Total documents: {len(loaded_docs)} | Total chunks: {len(all_chunks)}")
            else:
                print("No new chunks were added.")
            continue

        # Standard RAG question answering
        print("\n🔍 Retrieving relevant sections and generating response...")
        try:
            relevant_chunks = retriever.search(query, top_k=args.top_k)
            response = generator.generate(
                query=query,
                context=relevant_chunks,
                temperature=args.temperature
            )
            print(f"\n🤖 Answer:\n{response}\n")
            print("-" * 40)
            print("📚 Relevant sources:")
            for i, chunk in enumerate(relevant_chunks, 1):
                source_name = chunk.get('source', 'Unknown')
                score = chunk.get('rerank_score', 0.0)
                print(f"  [{i}] Source: {source_name} (Score: {score:.3f})")
                print(f"      Snippet: {chunk['text'][:120].strip()}...")
            print("-" * 40)
        except Exception as e:
            print(f"\n❌ Error during generation: {e}\n")


if __name__ == "__main__":
    main()
