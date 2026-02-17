
# Streamlit Application Specification for RAG Q&A System

## 1. Application Overview

**Purpose:**  
The application aims to build a Retrieval-Augmented Generation (RAG) Q&A system focused on financial documents. It allows users to upload or select financial documents, process and query them, and receive grounded, verifiable answers with source citations.

**High-level Story Flow:**  
1. **Upload Documents:** Users can upload financial documents for analysis.
2. **Chunk and Embed Documents:** The system preprocesses documents, splits them into chunks, and generates embeddings.
3. **Query Processing:** Users can input natural language queries to retrieve relevant information using semantic search.
4. **Answer Generation:** The system uses LLM to generate answers based on retrieved information.
5. **Performance Metrics:** Compare RAG vs. No-RAG performance on various metrics.

## 2. Code Requirements

### Import Statement

```python
from source import *
```

### UI Interactions and Function Calls

- **Upload Documents:** Calls `load_financial_documents()`
- **Chunk and Embed Documents:** Uses `chunk_documents()` and `create_and_populate_vector_store()`
- **Query Input:** Processes queries using `retrieve()` and `rag_answer()`
- **Performance Evaluation:** Utilizes `evaluate_qa()` for RAG vs No-RAG comparison

### Session State Design

- **Initialization:**  
  ```python
  if "documents" not in st.session_state:
      st.session_state["documents"] = None
  if "chunks" not in st.session_state:
      st.session_state["chunks"] = None
  if "query_results" not in st.session_state:
      st.session_state["query_results"] = None
  ```

- **Updating:**  
  - After document upload and processing, update `st.session_state["documents"]` and `st.session_state["chunks"]`.

- **Reading:**  
  - Access `st.session_state["documents"]` and `st.session_state["chunks"]` to generate answers.

### Markdown

- **Introduction:**
  ```python
  st.markdown("### RAG Q&A System for Financial Documents")
  st.markdown("This application demonstrates a pipeline for creating a grounded, hallucination-resistant Q&A system using financial documents.")
  ```

- **Upload Section:**
  ```python
  st.markdown("### Upload Financial Documents")
  st.markdown("Upload or select your financial documents in PDF format to start processing.")
  ```

- **Chunking and Embedding:**
  ```python
  st.markdown("### Document Chunking and Embedding")
  st.markdown("Documents are split into chunks and converted into embeddings for efficient retrieval.")
  ```

- **Query Section:**
  ```python
  st.markdown("### Query the Document")
  st.markdown("Input your query to retrieve relevant information and generate answers with citations.")
  ```

- **Performance Metrics:**
  ```python
  st.markdown("### RAG vs No-RAG Performance")
  st.markdown("Compare the effectiveness of the RAG system vs using an LLM without retrieval.")
  ```

- **Mathematical Formulations:**
  ```python
  st.markdown(r"$$\text{Answer Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Questions}}$$")
  st.markdown(r"where the accuracy depicts the percentage of correct answers.")
  ```

  ```python
  st.markdown(r"$$\text{Hallucination Rate} = \frac{\text{Number of Incorrect Facts}}{\text{Total Answers} - \text{IDK Responses}}$$")
  st.markdown(r"indicating the rate of incorrect information generated.")
  ```

  ```python
  st.markdown(r"$$\text{Retrieval Quality} \propto \frac{\text{I(c)}}{\text{|c|}} \times \text{Context Completeness}$$")
  st.markdown(r"illustrating the balance between information density and context completeness.")
  ```

## Application Layout and Interaction Flow

1. **Sidebar Navigation:**
   - "Upload Documents"
   - "Chunk & Embed"
   - "Query the System"
   - "Performance Metrics"

2. **Main Page:**
   - Display dynamic content based on sidebar selection.

3. **Widgets:**
   - File uploader on "Upload Documents" page.
   - Text input for queries on "Query the System" page.
   - Buttons to execute specific actions (e.g., processing documents, evaluating queries).

4. **Visualization:**
   - Display bar chart and line plots for performance metrics.
```
