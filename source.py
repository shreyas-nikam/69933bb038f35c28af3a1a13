import os
import fitz # PyMuPDF for PDF parsing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import tiktoken
import re
import requests # For downloading sample documents

# Global constant for the RAG system prompt
RAG_SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions ONLY using the provided document excerpts.\n\nSTRICT RULES:\n1. Answer ONLY from the provided context. Do NOT use your training data.\n2. If the answer is not in the provided context, respond: "Information not available in the provided documents."\n3. Quote specific numbers exactly as they appear in the context.\n4. After each factual claim, cite the source as [Source: company_period_chunk_id]. For example, [Source: AAPL_Earnings_AAPL_0_1].\n5. If multiple chunks contain relevant information, synthesize them and cite all sources."""

class FinancialRAGSystem:
    """
    A class encapsulating the RAG pipeline for financial document analysis.
    It handles document loading, chunking, embedding, vector store management,
    and RAG-based question answering.
    """

    def __init__(self, doc_dir='financial_docs', collection_name="financial_docs_collection",
                 chunk_size=500, chunk_overlap=100, llm_model='gpt-4o',
                 embedding_model_name='all-MiniLM-L6-v2', openai_api_key=None):
        """
        Initializes the FinancialRAGSystem.

        Args:
            doc_dir (str): Directory where financial PDF documents are stored.
            collection_name (str): Name for the ChromaDB collection.
            chunk_size (int): Target number of tokens per chunk for text splitting.
            chunk_overlap (int): Number of tokens of overlap between chunks.
            llm_model (str): The name of the OpenAI LLM model to use (e.g., 'gpt-4o').
            embedding_model_name (str): The name of the SentenceTransformer model for embeddings.
            openai_api_key (str, optional): Your OpenAI API key. If not provided,
                                            it will attempt to read from the OPENAI_API_KEY
                                            environment variable.
        """
        self.doc_dir = doc_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name

        # Initialize OpenAI client
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please provide it via the constructor or environment variable.")
        self.client_llm = OpenAI(api_key=api_key)

        # Initialize tiktoken encoder for token counting
        self.enc = tiktoken.encoding_for_model(self.llm_model)

        # Initialize embedding model
        self.embedder = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB client (in-memory for this example)
        self.chroma_client = chromadb.Client()
        self.collection = None # This will be populated after documents are processed
        self.loaded_documents = [] # Stores loaded documents after parsing

    def _load_financial_documents(self):
        """
        Loads PDF documents from the specified directory, extracts full text,
        and extracts metadata from filenames. Creates a dummy PDF if no documents
        are found in the directory.

        Returns:
            list: A list of dictionaries, each containing 'text' and 'metadata'.
        """
        documents = []
        # Create the directory if it doesn't exist
        if not os.path.exists(self.doc_dir):
            print(f"Creating directory: {self.doc_dir}.")
            os.makedirs(self.doc_dir)

        pdf_files = [f for f in os.listdir(self.doc_dir) if f.endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF documents found in '{self.doc_dir}'. Creating a dummy PDF for demonstration.")
            dummy_filepath = os.path.join(self.doc_dir, "GLOBALHORIZONS_Q1_2024_Earnings.pdf")
            dummy_text = (
                "Global Horizons Investment Firm reports Q1 2024 revenue of $10.5 billion, "
                "an 8% increase year-over-year. Net income was $2.1 billion. "
                "Earnings per share (EPS) reached $1.25. The firm diversified its portfolio, "
                "with a 15% growth in its technology sector investments. Key risks "
                "include geopolitical instability and fluctuating interest rates. "
                "Management expects Q2 2024 revenue to be between $10.8 billion and $11.2 billion. "
                "The board has 9 independent directors out of 11 total."
            )
            try:
                with fitz.open() as doc:
                    page = doc.new_page()
                    page.insert_text(dummy_text)
                    doc.save(dummy_filepath)
                pdf_files.append(os.path.basename(dummy_filepath)) # Add dummy to list
            except Exception as e:
                print(f"Error creating dummy PDF: {e}")
                return []

        for filename in pdf_files:
            filepath = os.path.join(self.doc_dir, filename)
            full_text = ""
            doc = fitz.open(filepath)
            for page in doc:
                full_text += page.get_text()

            # Extract metadata from filename: COMPANY_DOCTYPE_PERIOD.pdf
            # Example: AAPL_Q4_2024_Earnings.pdf -> Company: AAPL, DocType: Q4_2024, Period: Earnings
            name_parts = filename.replace('.pdf', '').split('_')

            metadata = {
                'company': name_parts[0] if len(name_parts) > 0 else 'Unknown',
                'doc_type': '_'.join(name_parts[1:-1]) if len(name_parts) > 2 else 'Unknown',
                'period': name_parts[-1] if len(name_parts) > 1 else 'Unknown',
                'filename': filename,
                'n_pages': len(doc),
                'n_words': len(full_text.split()),
                'source': f"{name_parts[0]}_{name_parts[-1]}", # Simplified source for citation
            }

            documents.append({
                'text': full_text,
                'metadata': metadata
            })
            doc.close()

        print(f"Loaded {len(documents)} documents:")
        for d in documents:
            m = d['metadata']
            print(f" - {m['company']} {m['doc_type']} {m['period']}: {m['n_pages']} pages, {m['n_words']:,} words")
        self.loaded_documents = documents
        return documents

    def _chunk_documents(self, documents):
        """
        Splits documents into overlapping chunks for embedding, maintaining metadata.

        Args:
            documents (list): List of dictionaries, each with 'text' and 'metadata'.

        Returns:
            list: A list of dictionaries, each representing a chunk with its text and metadata.
        """
        all_chunks = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", ",", " ", ""],
            length_function=lambda t: len(self.enc.encode(t))
        )

        for doc_idx, doc in enumerate(documents):
            chunks_text = splitter.split_text(doc['text'])

            for i, chunk_text in enumerate(chunks_text):
                chunk_metadata = {
                    **doc['metadata'],
                    'chunk_id': f"{doc['metadata']['company']}_{doc_idx}_{i}",
                    'n_tokens': len(self.enc.encode(chunk_text)),
                }
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })

        print(f"Total chunks created: {len(all_chunks)}")
        avg_tokens = sum(c['metadata']['n_tokens'] for c in all_chunks) / len(all_chunks) if all_chunks else 0
        print(f"Average tokens per chunk: {avg_tokens:.0f}")
        return all_chunks

    def _create_and_populate_vector_store(self, chunks, collection_name=None):
        """
        Embeds document chunks and stores them in a ChromaDB vector store.

        Args:
            chunks (list): List of dictionaries, each with 'text' and 'metadata'.
            collection_name (str, optional): Override the default collection name for this operation.

        Returns:
            chromadb.api.models.Collection.Collection: The ChromaDB collection object.
        """
        current_collection_name = collection_name if collection_name else self.collection_name

        # Create a new collection or get an existing one
        try:
            # Check if collection exists
            self.chroma_client.get_collection(name=current_collection_name)
            print(f"Collection '{current_collection_name}' already exists. Clearing it for fresh data.")
            self.chroma_client.delete_collection(name=current_collection_name)
            self.collection = self.chroma_client.create_collection(name=current_collection_name, metadata={"hnsw:space": "cosine"})
        except Exception: # If collection doesn't exist, create it
            print(f"Creating new collection: {current_collection_name}")
            self.collection = self.chroma_client.create_collection(name=current_collection_name, metadata={"hnsw:space": "cosine"})

        if not chunks:
            print("No chunks to add to the vector store.")
            return self.collection

        # Prepare data for ChromaDB
        texts = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        ids = [c['metadata']['chunk_id'] for c in chunks]

        # Generate embeddings in batches
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=64)
        print(f"Embeddings shape: {embeddings.shape}")

        # Add to ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(), # ChromaDB expects list of lists
            metadatas=metadatas,
            ids=ids
        )

        print(f"Vector store: {self.collection.count()} chunks indexed in '{current_collection_name}'")
        return self.collection

    def initialize_rag_pipeline(self):
        """
        Orchestrates the entire RAG pipeline setup:
        1. Loads financial documents.
        2. Chunks the documents.
        3. Creates and populates the ChromaDB vector store.
        """
        documents = self._load_financial_documents()
        chunks = self._chunk_documents(documents)
        self._create_and_populate_vector_store(chunks)
        print("RAG pipeline initialized successfully.")

    def retrieve(self, query, k=5):
        """
        Retrieves the top-k most relevant chunks for a given query from the vector store.

        Args:
            query (str): The natural language query.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            list: A list of dictionaries, each representing a retrieved chunk
                  with its text, metadata, and similarity score.
        """
        if not self.collection:
            raise RuntimeError("RAG pipeline not initialized. Call initialize_rag_pipeline() first.")

        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        retrieved_chunks = []
        if results['documents'] and results['distances'] and results['metadatas']:
            for i in range(len(results['documents'][0])):
                retrieved_chunks.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i] # ChromaDB distance is L2, convert to cosine sim
                })
        return retrieved_chunks

    def rag_answer(self, query, k=5, temperature=0.0, max_tokens=500):
        """
        Full RAG pipeline: retrieve -> generate with citations.

        Args:
            query (str): The natural language query.
            k (int): Number of chunks to retrieve.
            temperature (float): LLM generation temperature.
            max_tokens (int): Max tokens for LLM response.

        Returns:
            dict: A dictionary containing the query, generated answer, sources,
                  top similarity, input tokens, and output tokens.
        """
        retrieved = self.retrieve(query, k=k)

        # Format context for the LLM
        context = ""
        for i, r in enumerate(retrieved):
            # Ensure 'chunk_id' is processed safely, especially if it might be missing or malformed
            chunk_id_suffix = r['metadata'].get('chunk_id', 'unknown_chunk').split('_')[-1]
            context += f"\n--- Source: {r['metadata']['company']}_{r['metadata']['period']}_chunk_{chunk_id_suffix} ---\n"
            context += r['text'] + "\n"

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        try:
            response = self.client_llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content
            usage = response.usage
        except Exception as e:
            print(f"Error calling LLM for RAG: {e}")
            answer = "Error generating answer."
            usage = None

        # Extract source metadata for the return object
        source_metadatas = [r['metadata'] for r in retrieved]
        top_similarity = retrieved[0]['similarity'] if retrieved else 0.0

        return {
            'query': query,
            'answer': answer,
            'sources': source_metadatas,
            'top_similarity': top_similarity,
            'input_tokens': usage.prompt_tokens if usage else 0,
            'output_tokens': usage.completion_tokens if usage else 0,
        }

    def no_rag_answer(self, query, temperature=0.0, max_tokens=500):
        """
        Generates an answer WITHOUT retrieval (using LLM's parametric memory).

        Args:
            query (str): The natural language query.
            temperature (float): LLM generation temperature.
            max_tokens (int): Max tokens for LLM response.

        Returns:
            dict: A dictionary containing the query, generated answer, and usage info.
        """
        messages = [
            {"role": "system", "content": "You are a financial analyst. Answer questions about companies based on your knowledge."},
            {"role": "user", "content": query}
        ]

        try:
            response = self.client_llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content
            usage = response.usage
        except Exception as e:
            print(f"Error calling LLM for no-RAG: {e}")
            answer = "Error generating answer (No-RAG)."
            usage = None

        return {
            'query': query,
            'answer': answer,
            'input_tokens': usage.prompt_tokens if usage else 0,
            'output_tokens': usage.completion_tokens if usage else 0,
        }

    def expand_query(self, query, temperature=0.7, max_tokens=200):
        """
        Uses an LLM to generate alternative phrasings for a given query.

        Args:
            query (str): The original query.
            temperature (float): LLM generation temperature for creativity.
            max_tokens (int): Max tokens for LLM response.

        Returns:
            list: A list containing the original query and its alternative phrasings.
        """
        messages = [
            {"role": "user", "content": f"Generate 3 alternative phrasings of this financial question. Return only the alternatives, one per line.\nOriginal: {query}"}
        ]
        try:
            response = self.client_llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            alternatives_raw = response.choices[0].message.content.strip()
            alternatives = alternatives_raw.split('\n')
            return [query] + [alt for alt in alternatives if alt.strip()] # Filter empty lines
        except Exception as e:
            print(f"Error expanding query: {e}")
            return [query] # Return original query if expansion fails

    def multi_hop_rag(self, query, k=3, hops=2):
        """
        Performs multi-hop RAG by iteratively retrieving and refining the query.

        Args:
            query (str): The initial query.
            k (int): Number of chunks per retrieval step.
            hops (int): Number of retrieval-generation iterations.

        Returns:
            str: The final generated answer.
        """
        if not self.collection:
            raise RuntimeError("RAG pipeline not initialized. Call initialize_rag_pipeline() first.")

        context_chunks = []
        current_query = query

        for hop in range(hops):
            print(f"--- Hop {hop + 1}: Query: '{current_query}' ---")
            retrieved = self.retrieve(current_query, k=k)
            context_chunks.extend(retrieved)

            if hop < hops - 1: # If not the last hop, generate intermediate answer to refine query
                intermediate_context = ""
                for r in retrieved:
                    chunk_id_suffix = r['metadata'].get('chunk_id', 'unknown_chunk').split('_')[-1]
                    intermediate_context += f"\n--- Source: {r['metadata']['company']}_{r['metadata']['period']}_chunk_{chunk_id_suffix} ---\n"
                    intermediate_context += r['text'] + "\n"

                messages = [
                    {"role": "system", "content": "You are a financial analyst assistant. Extract key information from the provided context relevant to the query to refine the next query. Do not provide a full answer."},
                    {"role": "user", "content": f"Context:\n{intermediate_context}\n\nOriginal Question: {query}\n\nTask: Extract facts from the context to refine the next retrieval query related to the original question. If the information is not present, state so. Limit to 100 tokens."}
                ]

                try:
                    response = self.client_llm.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=100,
                    )
                    intermediate_answer = response.choices[0].message.content
                    print(f"Intermediate thought: {intermediate_answer}")
                    # Refine the query for the next hop
                    current_query = f"{query} Additional context for refinement: {intermediate_answer}"
                except Exception as e:
                    print(f"Error in intermediate generation: {e}")
                    break # Exit if LLM call fails

        # Final generation with all accumulated context
        final_context = ""
        # Deduplicate context chunks by chunk_id before final generation
        seen_chunk_ids = set()
        unique_context_chunks = []
        for r in context_chunks:
            chunk_id = r['metadata'].get('chunk_id')
            if chunk_id not in seen_chunk_ids:
                unique_context_chunks.append(r)
                seen_chunk_ids.add(chunk_id)

        for r in unique_context_chunks:
            chunk_id_suffix = r['metadata'].get('chunk_id', 'unknown_chunk').split('_')[-1]
            final_context += f"\n--- Source: {r['metadata']['company']}_{r['metadata']['period']}_chunk_{chunk_id_suffix} ---\n"
            final_context += r['text'] + "\n"

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{final_context}\n\nQuestion: {query}"}
        ]

        try:
            response = self.client_llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in multi-hop RAG final generation: {e}")
            return "Error in multi-hop RAG final generation."

    def evaluate_qa(self, qa_set, k=5):
        """
        Evaluates RAG and No-RAG performance on a given set of Q&A pairs.

        Args:
            qa_set (list): List of dicts, each with 'query' and 'ground_truth'.
            k (int): Number of chunks for RAG retrieval.

        Returns:
            pd.DataFrame: DataFrame with evaluation results for each query.
        """
        results_comparison = []

        for item in qa_set:
            query = item['query']
            ground_truth = str(item['ground_truth']).lower()

            # RAG Answer
            rag_res = self.rag_answer(query, k=k, model=self.llm_model)
            rag_answer_text = rag_res['answer'].lower()

            # No-RAG Answer
            norag_res = self.no_rag_answer(query, model=self.llm_model)
            norag_answer_text = norag_res['answer'].lower()

            # Evaluation Metrics (simplified for demonstration)
            rag_correct = ground_truth in rag_answer_text
            norag_correct = ground_truth in norag_answer_text

            rag_idk = "information not available" in rag_answer_text
            norag_idk = "information not available" in norag_answer_text or "not found" in norag_answer_text

            rag_has_citation = bool(re.search(r"\[source:\s*[\w_]+_chunk_\d+\]", rag_answer_text))

            retrieved_chunks_for_recall = self.retrieve(query, k=k)
            retrieved_texts = [r['text'].lower() for r in retrieved_chunks_for_recall]
            retrieval_recall_k = any(ground_truth in text for text in retrieved_texts)

            results_comparison.append({
                'query': query,
                'ground_truth': item['ground_truth'],
                'rag_answer': rag_res['answer'],
                'norag_answer': norag_res['answer'],
                'rag_correct': rag_correct,
                'norag_correct': norag_correct,
                'rag_idk': rag_idk,
                'norag_idk': norag_idk,
                'rag_has_citation': rag_has_citation,
                'retrieval_recall_k': retrieval_recall_k,
                'rag_top_similarity': rag_res['top_similarity'],
            })

        return pd.DataFrame(results_comparison)

    def evaluate_chunk_size(self, evaluation_set, k=5, chunk_sizes=None):
        """
        Evaluates RAG accuracy across different chunk sizes.

        Args:
            evaluation_set (list): Q&A pairs for evaluation.
            k (int): Number of chunks for RAG retrieval.
            chunk_sizes (list, optional): List of chunk sizes (in tokens) to test.
                                           Defaults to [200, 400, 600, 800, 1000].

        Returns:
            pd.DataFrame: A DataFrame with chunk size and corresponding RAG accuracy.
        """
        if chunk_sizes is None:
            chunk_sizes = [200, 400, 600, 800, 1000]

        results = []
        original_chunk_size = self.chunk_size
        original_chunk_overlap = self.chunk_overlap
        original_collection = self.collection # Store current collection

        if not self.loaded_documents:
            print("No documents loaded. Loading them for chunk size evaluation.")
            self._load_financial_documents() # Ensure documents are loaded

        for cs in chunk_sizes:
            print(f"\n--- Evaluating with Chunk Size: {cs} tokens ---")
            self.chunk_size = cs
            self.chunk_overlap = int(cs * 0.1) # Maintain ~10% overlap

            # 1. Chunk documents with current chunk size
            current_chunks = self._chunk_documents(self.loaded_documents)

            # 2. Create and populate vector store for current chunks with a temporary collection name
            temp_collection_name = f"{self.collection_name}_cs{cs}"
            self._create_and_populate_vector_store(current_chunks, collection_name=temp_collection_name)

            # 3. Evaluate RAG performance using the temporary collection
            # Need to temporarily set self.collection for retrieve/rag_answer to use it
            temp_collection_ref = self.collection
            self.collection = self.chroma_client.get_collection(name=temp_collection_name) # Ensure we use the current temp collection

            comp_df = self.evaluate_qa(evaluation_set, k=k)
            rag_accuracy = comp_df['rag_correct'].mean() * 100

            results.append({'chunk_size': cs, 'rag_accuracy': rag_accuracy})
            print(f"RAG Accuracy for chunk_size {cs}: {rag_accuracy:.1f}%")

            # Clean up temporary collection
            try:
                self.chroma_client.delete_collection(name=temp_collection_name)
                print(f"Cleaned up temporary collection: {temp_collection_name}")
            except Exception as e:
                print(f"Error cleaning up collection {temp_collection_name}: {e}")

            self.collection = temp_collection_ref # Restore collection reference for next iteration

        # Restore original chunking parameters and collection
        self.chunk_size = original_chunk_size
        self.chunk_overlap = original_chunk_overlap
        self.collection = original_collection # Restore original collection

        return pd.DataFrame(results)

def _download_sample_docs(doc_dir='financial_docs', urls=None):
    """
    Helper function to download sample PDF documents if they don't exist.
    This mimics the `!wget` commands from the notebook.
    """
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)

    if urls is None:
        urls = [
            "https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+28/TSLA_Q4_2025_Earnings.pdf",
            "https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+28/IR_Q4_2025_Earnings.pdf",
            "https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+28/GOOGL_Q4_2025_Earnings.pdf",
            "https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+28/AAPL_Q4_2025_Earnings.pdf"
        ]

    for url in urls:
        filename = os.path.join(doc_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {os.path.basename(url)} to {doc_dir}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {os.path.basename(url)}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
        else:
            print(f"{os.path.basename(url)} already exists in {doc_dir}.")

def main():
    """
    Main function to demonstrate the FinancialRAGSystem capabilities.
    This function will be executed when the script is run directly.
    """
    # Set OpenAI API key from environment variable or replace "YOUR_OPENAI_API_KEY"
    # It's recommended to set it as an environment variable (e.g., OPENAI_API_KEY)
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    # Download sample documents (if not present)
    _download_sample_docs()

    # Initialize the RAG system
    rag_system = FinancialRAGSystem(
        doc_dir='financial_docs',
        collection_name="financial_docs_main_collection",
        chunk_size=500,
        chunk_overlap=100,
        llm_model='gpt-4o',
        embedding_model_name='all-MiniLM-L6-v2'
        # openai_api_key="YOUR_OPENAI_API_KEY" # Uncomment and replace if not using env variable
    )

    # Initialize the RAG pipeline (load, chunk, embed, populate vector store)
    rag_system.initialize_rag_pipeline()

    # --- Demonstrate RAG Q&A ---
    print("\n--- Demonstrating RAG Q&A ---")
    test_queries = [
        "What was AAPL's total net sales?",
        "How was the quarter for TSLA?",
        "What is the Underlying performance for IR?",
        "What was the Alphabet CEO's total compensation in Q4 2025?",
        "What was Global Horizons' net income in Q1 2024?",
    ]

    for query in test_queries[:4]: # Run first 4 queries for demonstration
        result = rag_system.rag_answer(query)
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"Top Similarity (retrieved): {result['top_similarity']:.3f}")
        print("-" * 70)

    # Demonstrate the 'I don't know' guardrail for a query where info is not available
    print("\n--- Testing 'I don't know' guardrail ---")
    result_idk = rag_system.rag_answer(test_queries[3]) # Alphabet CEO compensation
    print(f"\nQ: {result_idk['query']}")
    print(f"A: {result_idk['answer']}")
    print(f"Top Similarity (retrieved): {result_idk['top_similarity']:.3f}")
    print("-" * 70)
    
    # Demonstrate a query for the dummy document
    print("\n--- Testing query on dummy document ---")
    result_dummy = rag_system.rag_answer(test_queries[4])
    print(f"\nQ: {result_dummy['query']}")
    print(f"A: {result_dummy['answer']}")
    print(f"Top Similarity (retrieved): {result_dummy['top_similarity']:.3f}")
    print("-" * 70)


    # --- Define an evaluation set with ground truth ---
    evaluation_set = [
        # Apple (AAPL Q4 FY2025)
        {"query": "What were Apple’s total net sales for the three months ended September 27, 2025?", "ground_truth": "$102,466 million"},
        {"query": "What was Apple’s net income for the three months ended September 27, 2025?", "ground_truth": "$27,466 million"},
        {"query": "What was Apple’s diluted earnings per share (EPS) for the three months ended September 27, 2025?", "ground_truth": "$1.85"},
        {"query": "What was Apple’s cash and cash equivalents as of September 27, 2025?", "ground_truth": "$35,934 million"},

        # Alphabet (Q4 2025 / FY 2025 results)
        {"query": "What were Alphabet’s total revenues for the quarter ended December 31, 2025?", "ground_truth": "$113,828 million"},
        {"query": "What was Alphabet’s net income for the quarter ended December 31, 2025?", "ground_truth": "$34,455 million"},
        {"query": "What was Alphabet’s diluted net income per share for the quarter ended December 31, 2025?", "ground_truth": "$2.82"},
        {"query": "What were Google Cloud revenues for the quarter ended December 31, 2025?", "ground_truth": "$17,664 million"},

        # IR (Q4 2025)
        {"query": "What was Ingersoll Rand's total revenue for Q4 2025?", "ground_truth": "$1.74 billion"},
        {"query": "What was Ingersoll Rand's adjusted EBITDA for Q4 2025?", "ground_truth": "$430.7 million"},
        {"query": "What was Ingersoll Rand's net income for Q4 2025?", "ground_truth": "$166.7 million"},

        # TSLA (Q4 2025)
        {"query": "What were Tesla's total revenues for the three months ended December 31, 2025?", "ground_truth": "$28,510 million"},
        {"query": "What was Tesla's net income for the three months ended December 31, 2025?", "ground_truth": "$7,983 million"},
        {"query": "What was Tesla's free cash flow for the three months ended December 31, 2025?", "ground_truth": "$3,280 million"},

        # Global Horizons (Dummy)
        {"query": "What was Global Horizons Investment Firm's Q1 2024 revenue?", "ground_truth": "$10.5 billion"},
        {"query": "How many independent directors does Global Horizons' board have?", "ground_truth": "9"},
        {"query": "What are the key risks mentioned for Global Horizons?", "ground_truth": "geopolitical instability and fluctuating interest rates"},

        # IDK / not-stated test
        {"query": "What was Unilever’s CEO’s base salary for 2025?", "ground_truth": "Information not available"},
        {"query": "What is the capital expenditure budget for Apple in 2026?", "ground_truth": "Information not available"},
    ]

    print("\n--- RAG vs. No-RAG COMPARISON ---")
    comp_df = rag_system.evaluate_qa(evaluation_set, k=5)
    print("=" * 50)
    print(f"RAG Accuracy: {comp_df['rag_correct'].mean() * 100:.1f}%")
    print(f"No-RAG Accuracy: {comp_df['norag_correct'].mean() * 100:.1f}%")
    print(f"RAG Hallucination Rate: {((~comp_df['rag_correct']) & (~comp_df['rag_idk'])).mean() * 100:.1f}%")
    print(f"No-RAG Hallucination Rate: {((~comp_df['norag_correct']) & (~comp_df['norag_idk'])).mean() * 100:.1f}%")
    print(f"RAG Citation Rate: {comp_df['rag_has_citation'].mean() * 100:.1f}%")
    print(f"RAG 'I Don't Know' Rate: {comp_df['rag_idk'].mean() * 100:.1f}%")
    print(f"No-RAG 'I Don't Know' Rate: {comp_df['norag_idk'].mean() * 100:.1f}%")
    print(f"RAG Retrieval Recall@k: {comp_df['retrieval_recall_k'].mean() * 100:.1f}%")

    # The visualization code is kept here but `plt.show()` calls are commented out
    # as they are typically not run in an `app.py` directly, but the data
    # for plots might be generated and sent to a frontend.

    # Visualization: Bar Chart RAG vs No-RAG Performance
    metrics_data = {
        'Metric': ['Answer Accuracy', 'Hallucination Rate', 'Citation Rate', "'I Don't Know' Rate"],
        'RAG': [
            comp_df['rag_correct'].mean() * 100,
            ((~comp_df['rag_correct']) & (~comp_df['rag_idk'])).mean() * 100,
            comp_df['rag_has_citation'].mean() * 100,
            comp_df['rag_idk'].mean() * 100
        ],
        'No-RAG': [
            comp_df['norag_correct'].mean() * 100,
            ((~comp_df['norag_correct']) & (~comp_df['norag_idk'])).mean() * 100,
            0, # No-RAG has no citations by definition
            comp_df['norag_idk'].mean() * 100
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df_melted = metrics_df.melt('Metric', var_name='Approach', value_name='Value')

    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='Metric', y='Value', hue='Approach', data=metrics_df_melted)
    # plt.title('RAG vs. No-RAG Performance Comparison for Financial Q&A')
    # plt.ylabel('Percentage (%)')
    # plt.ylim(0, 100)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    # Visualization: Histogram of Top-1 Similarity Scores
    # plt.figure(figsize=(10, 5))
    # sns.histplot(comp_df['rag_top_similarity'], bins=10, kde=True)
    # plt.title('Distribution of Top-1 Retrieval Similarity Scores')
    # plt.xlabel('Cosine Similarity Score')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    print("\n--- Example Q&A Gallery with Ground-Truth Verification (first 3 examples) ---")
    for i, row in comp_df.head(3).iterrows(): # Display first 3 examples
        print(f"\nQuery: {row['query']}")
        print(f"Ground Truth: {row['ground_truth']}")
        print(f"RAG Answer: {row['rag_answer']} (Correct: {row['rag_correct']}, Cited: {row['rag_has_citation']})")
        print(f"No-RAG Answer: {row['norag_answer']} (Correct: {row['norag_correct']})")
        print("-" * 70)

    # --- Demonstrate Query Expansion ---
    print("\n--- Demonstrating Query Expansion ---")
    original_query_mismatch = "What was Global Horizons' cash burn rate in Q1 2024?"
    print(f"Original Query: {original_query_mismatch}")
    expanded_queries = rag_system.expand_query(original_query_mismatch)
    print(f"Expanded Queries: {expanded_queries}")

    print("\nRetrieving with Original Query:")
    res_orig = rag_system.retrieve(original_query_mismatch, k=2)
    for r in res_orig:
        print(f"  - [{r['similarity']:.3f}] Chunk ID: {r['metadata']['chunk_id']}, Text: {r['text'][:100]}...")

    print("\nRetrieving with Expanded Query (simulated, e.g., 'operating cash consumption'):")
    # For a full implementation, you'd run retrieve for each expanded query and combine results.
    # Here, we simulate by using one better-phrased query.
    simulated_better_query = "What was Global Horizons' operating cash consumption in Q1 2024?"
    res_expanded = rag_system.retrieve(simulated_better_query, k=2)
    for r in res_expanded:
        print(f"  - [{r['similarity']:.3f}] Chunk ID: {r['metadata']['chunk_id']}, Text: {r['text'][:100]}...")


    # --- Demonstrate Multi-Hop RAG (Conceptual) ---
    print("\n--- Demonstrating Multi-Hop RAG (Conceptual) ---")
    multi_hop_query = "Summarize Global Horizons' financial performance and future outlook mentioned, considering both revenue growth and key risks."
    print(f"Multi-Hop Query: {multi_hop_query}")
    multi_hop_answer = rag_system.multi_hop_rag(multi_hop_query, hops=2, k=3)
    print(f"\nFinal Multi-Hop Answer: {multi_hop_answer}")

    # --- Chunk size sensitivity analysis ---
    print("\n--- Starting Chunk Size Sensitivity Analysis ---")
    chunk_size_results = rag_system.evaluate_chunk_size(evaluation_set)

    print("\n--- Optimal Chunk Size Analysis Complete ---")
    print(chunk_size_results)

    # Visualization: Accuracy vs. Chunk Size
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x='chunk_size', y='rag_accuracy', data=chunk_size_results, marker='o')
    # plt.title('RAG Accuracy vs. Document Chunk Size')
    # plt.xlabel('Chunk Size (Tokens)')
    # plt.ylabel('RAG Answer Accuracy (%)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xticks(chunk_size_results['chunk_size'])
    # plt.ylim(0, 100)
    # plt.show()


if __name__ == "__main__":
    main()
