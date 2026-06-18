# Agentic RAG for Enterprise Knowledge Work

## Abstract
This demo paper discusses an Agentic RAG workflow for enterprise knowledge management. The system combines document ingestion, semantic retrieval, tool use, memory, and grounded answer generation.

## Method
The workflow first splits documents into chunks, stores them in a vector database, retrieves top-k relevant chunks for a user question, and asks a language model to answer with source evidence.

## Contributions
- Defines a practical Agentic RAG pipeline for knowledge-intensive tasks.
- Shows how memory and metadata filtering improve multi-turn user experience.
- Emphasizes observability and evaluation for production use.

## Limitations
The system still depends on retrieval quality and may fail when the knowledge base lacks relevant evidence.
