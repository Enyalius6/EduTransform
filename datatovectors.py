# datatovector.py

import pickle
import faiss
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
import spacy

def process_pdf(pdf_path):
    # Download and load NLP tools
    nltk.download('punkt')
    nlp = spacy.load("en_core_web_sm")
    # Initialize Sentence-BERT model for embeddings
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    def extract_text_and_tables(pdf_path):
        text_pages = []
        table_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract page text
                text = page.extract_text()
                if text:
                    text_pages.append({"page": i, "text": text})
                
                # Extract tables (if any)
                tables = page.extract_tables()
                if tables:
                    table_text = ""
                    for table in tables:
                        for row in table:
                            # Convert row (ignoring None values) into a string line
                            row_text = " | ".join([str(cell) for cell in row if cell is not None])
                            table_text += row_text + "\n"
                    if table_text.strip():
                        table_data.append({"page": i, "text": table_text})
        return text_pages, table_data

    def adaptive_chunking(text, max_chunk_size=1024, overlap=50):
        doc = nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            sentence_length = len(sentence_text.split())
            if current_length + sentence_length > max_chunk_size:
                merged_chunk = " ".join(current_chunk)
                if chunks and len(merged_chunk.split()) < (max_chunk_size // 2):
                    chunks[-1] += " " + merged_chunk
                else:
                    chunks.append(merged_chunk)
                # Retain overlap from the previous chunk
                current_chunk = current_chunk[-overlap:]
                current_length = sum(len(chunk.split()) for chunk in current_chunk)
            current_chunk.append(sentence_text)
            current_length += sentence_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def encode_and_store_in_faiss(documents):
        chunked_documents = []
        for doc in documents:
            chunks = adaptive_chunking(doc['text'])
            for chunk in chunks:
                chunked_documents.append({"text": chunk, "page": doc["page"]})
        # Prepare texts for embedding
        chunk_texts = [d["text"] for d in chunked_documents]
        document_embeddings = sbert_model.encode(chunk_texts, convert_to_tensor=False)
        document_embeddings = np.array(document_embeddings)
        # Normalize embeddings for cosine similarity
        document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        dimension = document_embeddings.shape[1]
        # Create a FAISS index (using HNSW)
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(document_embeddings)
        # Save the index and documents
        with open('faiss_index.pkl', 'wb') as f:
            pickle.dump(index, f)
        with open('documents.pkl', 'wb') as f:
            pickle.dump(chunked_documents, f)
        print("FAISS index and documents saved successfully!")

    # Extract text and tables from the given PDF
    text_pages, table_data = extract_text_and_tables(pdf_path)
    documents = text_pages + table_data
    # Process and store embeddings in FAISS
    encode_and_store_in_faiss(documents)
