# Paper Reading Assistant with Long-Term Memory

## Abstract
This demo paper describes a research reading assistant that extracts paper metadata, summarizes research questions, and stores important methods as long-term memory.

## Method
The assistant parses PDF or Markdown text, asks an LLM to produce structured JSON, stores paper metadata in SQLite, and writes important keywords into a memory database.

## Findings
Persistent conversation history helps users continue literature review tasks across sessions. Structured decomposition makes comparison across papers easier.

## Future Work
Future versions can add citation graph analysis, automatic benchmark evaluation, and multi-agent review.
