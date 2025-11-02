# app.py
import os
import tempfile
import pickle
import asyncio
import inspect
from typing import Any, List, Tuple

import streamlit as st
from openai import OpenAI

# LangChain / vectorstore imports
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------------- Embeddings import with fallbacks ----------------
# Try to import HuggingFaceEmbeddings from multiple possible packages so this app
# works across common LangChain setups.
EMB_IMPORT_SOURCE = None
try:
    # recommended partner package
    from langchain_huggingface import HuggingFaceEmbeddings
    EMB_IMPORT_SOURCE = "langchain-huggingface"
except Exception:
    try:
        # community wrapper (some environments)
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        EMB_IMPORT_SOURCE = "langchain_community"
    except Exception:
        try:
            # sometimes exposed by core langchain package (version dependent)
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
            EMB_IMPORT_SOURCE = "langchain"
        except Exception as e:
            raise ImportError(
                "HuggingFaceEmbeddings not found. Install `langchain-huggingface` or `langchain_community`."
            ) from e

# Instantiate embeddings (change model/device if you need)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # model_kwargs={"device": "cpu"}  # uncomment & set to 'cuda' if you have GPU
)

# ---------------- OpenRouter / OpenAI client ----------------
# Load your API key securely from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", None)
if OPENROUTER_API_KEY is None:
    st.error("OPENROUTER_API_KEY not found in Streamlit secrets. Add it and restart.")
    st.stop()

# Initialize OpenRouter client with the secret API key
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

def get_openrouter_response(prompt: str) -> str:
    """
    Send prompt to OpenRouter / OpenAI-compatible client and return text content.
    Adjust as needed for different response formats.
    """
    completion = client.chat.completions.create(
        extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Local Doc AI"},
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}],
    )
    # defensive: try to extract the message text in common shapes
    try:
        return completion.choices[0].message.content
    except Exception:
        # fallback - try other common shape
        try:
            return completion.choices[0].text
        except Exception:
            return str(completion)

# ------------------ FILE LOADING / VECTOR STORE ------------------

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

def load_faiss_index(index_dir):
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# ------------------ Retriever compatibility helper ------------------

def fetch_relevant_documents(retriever: Any, query: str, k: int = 3) -> List[Any]:
    """
    Try multiple retriever APIs (sync and async) and common entrypoints.
    Returns a list of Documents (or whatever the retriever returns).
    This function does NOT raise if methods are missing; it returns an empty list
    and writes a Streamlit warning so the UI keeps running.
    """
    # candidate kwargs
    kwargs = {}
    # Many retrievers accept search_kwargs={'k': k} or k=k
    # We'll try both patterns when calling
    try_kwargs_variants = [
        {"k": k},
        {"search_kwargs": {"k": k}},
        {},
    ]

    # 1) Common sync API
    if hasattr(retriever, "get_relevant_documents"):
        for kv in try_kwargs_variants:
            try:
                return retriever.get_relevant_documents(query, **kv)
            except TypeError:
                # try without kwargs
                try:
                    return retriever.get_relevant_documents(query)
                except Exception:
                    continue
            except Exception:
                continue

    # 2) Older/alternate sync name
    if hasattr(retriever, "get_relevant_docs"):
        try:
            return retriever.get_relevant_docs(query)
        except Exception:
            pass

    # 3) Runnable / invoke style (sync)
    if hasattr(retriever, "invoke"):
        try:
            res = retriever.invoke(query)
            if isinstance(res, list):
                return res
            return res
        except Exception:
            pass

    if hasattr(retriever, "run"):
        try:
            res = retriever.run(query)
            if isinstance(res, list):
                return res
            return res
        except Exception:
            pass

    # 4) Async variants
    if hasattr(retriever, "aget_relevant_documents"):
        for kv in try_kwargs_variants:
            try:
                return asyncio.run(retriever.aget_relevant_documents(query, **kv))
            except TypeError:
                try:
                    return asyncio.run(retriever.aget_relevant_documents(query))
                except Exception:
                    continue
            except Exception:
                continue

    if hasattr(retriever, "ainvoke"):
        try:
            return asyncio.run(retriever.ainvoke(query))
        except Exception:
            pass

    if hasattr(retriever, "abatch"):
        try:
            return asyncio.run(retriever.abatch([query]))
        except Exception:
            pass

    # nothing matched
    st.warning(
        "Retriever does not support known retrieval methods (get_relevant_documents / run / invoke / aget_relevant_documents). "
        "Falling back to vector_store.search (if available)."
    )
    return []

def fallback_vectorstore_search(vector_store: Any, query: str, k: int = 3) -> List[Any]:
    """
    Use the backing vector_store similarity search if retriever methods aren't available.
    Returns a list of Documents (or an empty list).
    """
    if vector_store is None:
        return []
    # Try common vector store search APIs
    try:
        # returns list[Document]
        return vector_store.similarity_search(query, k=k)
    except Exception:
        pass
    try:
        # returns list[(Document, score)]
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        return [d for d, _score in docs_with_scores]
    except Exception:
        pass

    # nothing worked
    st.error("Fallback search on the vector store failed. Inspect vector_store type and APIs.")
    return []

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="📄 Surendar's AI App", page_icon="🤖", layout="wide")
st.title("🤖 Surendar's AI Assistant")

st.markdown(f"**Embedding import source:** `{EMB_IMPORT_SOURCE}`")

# Bot selection
bot_choice = st.radio(
    "Choose Bot Mode:",
    ["Document Q&A Bot", "Surendar Bot"],
    horizontal=True
)

if bot_choice == "Document Q&A Bot":
    st.markdown("**💡 Example Questions:**")
    st.markdown("- What is the summary of the uploaded document?")
    st.markdown("- List the key points from the PDF.")
    st.markdown("- What is the main conclusion in the data?")

    uploaded_files = st.file_uploader(
        "Upload PDF or CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
        docs = load_files(uploaded_files)
        if not docs:
            st.error("No documents were loaded from the uploaded files.")
        else:
            vector_store = create_vector_store(docs)
            # Try to create retriever; some FAISS wrapper versions expose as_retriever differently
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            except Exception:
                # try without kwargs
                try:
                    retriever = vector_store.as_retriever()
                except Exception:
                    retriever = None
                    st.warning("Could not create retriever object from vector_store; will use vector_store directly.")

            query = st.text_input("Ask a question about your documents:")
            if query:
                relevant_docs = []
                if retriever is not None:
                    relevant_docs = fetch_relevant_documents(retriever, query, k=3)

                # Fallback to vector_store similarity search if retriever gave nothing
                if (not relevant_docs) and vector_store is not None:
                    relevant_docs = fallback_vectorstore_search(vector_store, query, k=3)

                if not relevant_docs:
                    st.info("No relevant documents found for this query.")
                else:
                    # join the doc texts into a context
                    context_texts = "\n\n".join(
                        getattr(doc, "page_content", str(doc)) for doc in relevant_docs
                    )
                    prompt = f"Answer the question based on the following context:\n\n{context_texts}\n\nQuestion: {query}"
                    with st.spinner("Querying the model..."):
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

    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        try:
            vector_store = load_faiss_index(index_dir)
        except Exception as e:
            st.error(f"Failed to load FAISS index from '{index_dir}': {e}")
            vector_store = None

        retriever = None
        if vector_store is not None:
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            except Exception:
                try:
                    retriever = vector_store.as_retriever()
                except Exception:
                    retriever = None
                    st.warning("Could not create retriever from FAISS index; will try fallback search.")

        st.success(
            "👋 Welcome! You are now chatting with **Surendar's** personal RAG-based bot. "
            "Feel free to ask about my background, projects, skills, or career journey."
        )

        query = st.text_input("Ask me anything about Surendar:")
        if query:
            relevant_docs = []
            if retriever is not None:
                relevant_docs = fetch_relevant_documents(retriever, query, k=3)

            # fallback to vector store if retriever fails
            if (not relevant_docs) and vector_store is not None:
                relevant_docs = fallback_vectorstore_search(vector_store, query, k=3)

            if not relevant_docs:
                st.info("No relevant documents found in the FAISS index for this query.")
            else:
                context_texts = "\n\n".join(
                    getattr(doc, "page_content", str(doc)) for doc in relevant_docs
                )
                prompt = f"Answer the question based on Surendar's resume:\n\n{context_texts}\n\nQuestion: {query}"
                with st.spinner("Querying the model..."):
                    answer = get_openrouter_response(prompt)
                st.subheader("Surendar Bot's Answer:")
                st.write(answer)
    else:
        st.error("FAISS index not found. Please create one in 'faiss_index' folder.")

# End of file
