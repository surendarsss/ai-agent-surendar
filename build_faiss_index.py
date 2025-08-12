import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
# Paths
pdf_path = "data/Surendar_Resume.pdf"
csv_path = "data/about_surendar.csv"
index_dir = "faiss_index"

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents
docs = []
if os.path.exists(pdf_path):
    docs.extend(PyPDFLoader(pdf_path).load())
else:
    print(f"⚠️ PDF not found: {pdf_path}")

if os.path.exists(csv_path):
    docs.extend(CSVLoader(file_path=csv_path).load())
else:
    print(f"⚠️ CSV not found: {csv_path}")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Create FAISS index
vector_store = FAISS.from_documents(split_docs, embeddings)

# Save index in index.faiss + index.pkl format
vector_store.save_local(index_dir)
print(f"✅ FAISS index saved in '{index_dir}' with index.faiss & index.pkl")
