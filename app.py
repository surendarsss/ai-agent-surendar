import streamlit as st
import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
import pickle

# ------------------ SETTINGS ------------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]  # Replace with your OpenRouter API key
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------ FUNCTIONS ------------------

def load_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
        elif file.name.endswith(".csv"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            loader = CSVLoader(file_path=tmp_path)
            docs.extend(loader.load())
    return docs

def create_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

def get_openrouter_response(prompt):
    completion = client.chat.completions.create(
        extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Local Doc AI"},
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def load_faiss_index(index_dir):
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="📄 Surendar's AI App", page_icon="🤖", layout="wide")
st.title("🤖 Surendar's AI Assistant")

bot_choice = st.sidebar.radio("Choose Bot Mode:", ["Document Q&A Bot", "Surendar Bot"])

if bot_choice == "Document Q&A Bot":
    st.markdown("**💡 Example Questions:**")
    st.markdown("- What is the summary of the uploaded document?")
    st.markdown("- List the key points from the PDF.")
    st.markdown("- What is the main conclusion in the data?")

    uploaded_files = st.file_uploader("Upload PDF or CSV files", type=["pdf", "csv"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
        docs = load_files(uploaded_files)
        vector_store = create_vector_store(docs)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        query = st.text_input("Ask a question about your documents:")
        if query:
            relevant_docs = retriever.get_relevant_documents(query)
            context_texts = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"Answer the question based on the following context:\n\n{context_texts}\n\nQuestion: {query}"
            answer = get_openrouter_response(prompt)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.info("Please upload at least one PDF or CSV file to begin.")

elif bot_choice == "Surendar Bot":
    st.markdown("**💡 Example Questions:**")
    st.markdown("- Who is Surendar?")
    st.markdown("- What projects has Surendar worked on?")
    st.markdown("- What are Surendar's technical skills?")
    st.markdown("- Describe Surendar's educational background.")
    st.markdown("- What work experience does Surendar have?")
    st.markdown("- What programming languages does Surendar know?")

    index_dir = "faiss_index"  # Your existing FAISS index folder
    if os.path.exists(index_dir):
        vector_store = load_faiss_index(index_dir)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        st.success("👋 Welcome! You are now chatting with **Surendar's** personal RAG-based bot. "
           "Feel free to ask about my background, projects, skills, or career journey.")

        
        query = st.text_input("Ask me anything about Surendar:")
        if query:
            relevant_docs = retriever.get_relevant_documents(query)
            context_texts = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"Answer the question based on Surendar's resume:\n\n{context_texts}\n\nQuestion: {query}"
            answer = get_openrouter_response(prompt)
            st.subheader("Surendar Bot's Answer:")
            st.write(answer)
    else:
        st.error("FAISS index not found. Please create one in 'faiss_index' folder.")