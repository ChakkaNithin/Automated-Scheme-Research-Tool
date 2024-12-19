import streamlit as st
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to fetch PDF content from a URL
def fetch_pdf_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            reader = PdfReader("temp.pdf")
            content = ""
            for page in reader.pages:
                content += page.extract_text()
            print(f"Fetched content from {url}: {content[:100]}")  
            return content
        else:
            st.error(f"Failed to fetch PDF: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching PDF content: {str(e)}")
        return None

# Function to read uploaded PDF content
def read_uploaded_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
        print(f"Read content from uploaded PDF: {content[:100]}")
        return content
    except Exception as e:
        st.error(f"Error reading uploaded PDF: {str(e)}")
        return None

# Function to create embeddings
def create_embeddings(text):
    try:
        embedding = model.encode(text)
        print(f"Created embedding for text: {text[:50]}") 
        return embedding
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

# Function to create a FAISS index
def create_faiss_index(embeddings, documents, file_path="faiss_store_local.pkl"):
    dimension = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype(np.float32))
    
    # Save the index and documents
    with open(file_path, "wb") as f:
        pickle.dump((faiss_index, documents), f)
    print(f"FAISS index saved as {file_path}")

# Function to load the FAISS index
def load_faiss_index(file_path="faiss_store_local.pkl"):
    with open(file_path, "rb") as f:
        faiss_index, documents = pickle.load(f)
    return faiss_index, documents

# Function to query the FAISS index
def query_faiss_index(faiss_index, query_embedding, documents, k=3):
    D, I = faiss_index.search(np.array([query_embedding]).astype(np.float32), k)
    results = [(documents[i], D[0][j]) for j, i in enumerate(I[0])]
    return results

# Streamlit App
st.title("Automated Scheme Research Tool")
st.sidebar.header("Input PDF URLs or Upload Files")

# Input for URLs
url_input = st.sidebar.text_area("Enter PDF URLs:")

# File uploader for PDFs
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Button to process input
if st.sidebar.button("Process Documents"):
    documents = []
    embeddings = []

    # Process URLs
    if url_input:
        urls = url_input.split("\n")
        for url in urls:
            content = fetch_pdf_content(url.strip())
            if content:
                documents.append(content)
                embedding = create_embeddings(content)
                if embedding is not None:
                    embeddings.append(embedding)

    # Process uploaded PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            content = read_uploaded_pdf(uploaded_file)
            if content:
                documents.append(content)
                embedding = create_embeddings(content)
                if embedding is not None:
                    embeddings.append(embedding)

    # Create FAISS index
    if embeddings:
        create_faiss_index(embeddings, documents)
        st.success("Documents processed and FAISS index created!")
    else:
        st.error("No valid documents found to process.")

# Query input
query = st.text_input("Ask a question about the scheme(s):")
if query and st.button("Get Answer"):
    query_embedding = create_embeddings(query)
    try:
        faiss_index, documents = load_faiss_index()
        results = query_faiss_index(faiss_index, query_embedding, documents)

        st.write("Top Results:")
        for i, (doc, score) in enumerate(results):
            st.write(f"Result {i+1} (Score: {score}):")
            st.write(doc[:400])  # Show first 400 characters
            st.write("-" * 50)

    except FileNotFoundError:
        st.error("FAISS index not found. Please process URLs or upload PDFs first.")
