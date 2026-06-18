# Evaluating LLM Applications with Retrieval Evidence

## Abstract
This demo paper focuses on evaluating LLM applications that use retrieved context. Important metrics include answer relevance, faithfulness, citation accuracy, and robustness under empty-context conditions.

## Evaluation Design
A small benchmark can contain fixed questions, expected evidence, human scores, and generated answers. Each answer is evaluated for whether it is supported by retrieved chunks.

## Contributions
- Proposes a simple evaluation table for course projects.
- Connects RAG quality with source traceability.
- Highlights the need for failure handling when the model or vector database is unavailable.
