import sys
from pathlib import Path
import hashlib
import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.document_loader import load_all_documents
from src.rag_pipeline import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DOCUMENTS_DIR = PROJECT_ROOT / "Documents"
UPLOAD_DIR = PROJECT_ROOT / "uploaded_documents"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 0.4rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .status-uploaded {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-default {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .source-tag {
        font-size: 0.85rem;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: inline-block;
        margin: 0.2rem;
    }
    .retry-box {
        margin-top: 0.5rem;
        padding: 0.8rem 1rem;
        border-radius: 0.5rem;
        background-color: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# LOAD BASE VECTOR STORE
# ============================================================

@st.cache_resource
def load_base_vector_store():
    if not (VECTOR_DB_DIR / "index.faiss").exists():
        return None

    try:
        embeddings = load_embeddings()
        return FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading base vector store: {e}")
        return None


# ============================================================
# BUILD VECTOR STORE FROM UPLOADED DOCUMENTS (ISOLATED)
# ============================================================

def build_vector_store_from_uploads():
    """
    Build a fresh FAISS vector store strictly from UPLOAD_DIR.
    Ensures no leakage from base documents or previously deleted uploads.
    """
    if not UPLOAD_DIR.exists():
        return None, 0, []

    uploaded_docs = load_all_documents(UPLOAD_DIR)
    if not uploaded_docs:
        return None, 0, []

    documents = []
    doc_names = set()

    for item in uploaded_docs:
        text = item.get("text", "").strip()
        if not text:
            continue

        meta = item.get("metadata", {})
        if "source" in meta:
            doc_names.add(meta["source"])

        documents.append(
            Document(
                page_content=text,
                metadata=meta
            )
        )

    if not documents:
        return None, 0, []

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)
    if not chunks:
        return None, 0, []

    # Embed and build FAISS
    embeddings = load_embeddings()
    store = FAISS.from_documents(chunks, embeddings)

    return store, len(chunks), sorted(list(doc_names))


# ============================================================
# UPLOAD HELPERS & CACHE MANAGEMENT
# ============================================================

def get_uploaded_file_signature(files):
    if not files:
        return ""

    signature_data = []
    for file in files:
        file_bytes = file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        signature_data.append(f"{file.name}:{file_hash}")

    return "|".join(sorted(signature_data))


def purge_upload_directory():
    """Remove all previously uploaded files from disk."""
    if UPLOAD_DIR.exists():
        for existing_file in UPLOAD_DIR.iterdir():
            if existing_file.is_file():
                try:
                    existing_file.unlink()
                except Exception as e:
                    print(f"Error removing {existing_file}: {e}")


def save_uploaded_files(files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    purge_upload_directory()

    saved_names = []
    for file in files:
        file_path = UPLOAD_DIR / file.name
        file_path.write_bytes(file.getvalue())
        saved_names.append(file.name)

    return saved_names


def clear_everything():
    """Clear all caches, uploaded files on disk, and session state."""
    st.cache_data.clear()
    st.cache_resource.clear()
    purge_upload_directory()

    st.session_state.messages = []
    st.session_state.active_store = None
    st.session_state.upload_signature = ""
    st.session_state.active_docs = []
    st.session_state.chunk_count = 0
    st.session_state.last_user_prompt = ""
    st.session_state.retry_prompt = None


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_store" not in st.session_state:
    st.session_state.active_store = None

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = ""

if "active_docs" not in st.session_state:
    st.session_state.active_docs = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "retry_prompt" not in st.session_state:
    st.session_state.retry_prompt = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 IntelliAssist AI")
    st.caption("Retrieval-Augmented Document Assistant")
    st.divider()

    st.subheader("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files. The AI will answer questions exclusively from these files."
    )

    current_signature = get_uploaded_file_signature(uploaded_files)

    # When new files are uploaded or changed
    if uploaded_files and current_signature != st.session_state.upload_signature:
        with st.spinner("Processing & indexing uploaded documents..."):
            saved_names = save_uploaded_files(uploaded_files)
            store, count, doc_names = build_vector_store_from_uploads()
            
            st.session_state.active_store = store
            st.session_state.upload_signature = current_signature
            st.session_state.active_docs = doc_names
            st.session_state.chunk_count = count

        st.success(f"Indexed {len(uploaded_files)} file(s) ({count} chunks)")

    elif not uploaded_files and st.session_state.upload_signature != "":
        # User removed all uploaded files
        purge_upload_directory()
        st.session_state.active_store = None
        st.session_state.upload_signature = ""
        st.session_state.active_docs = []
        st.session_state.chunk_count = 0
        st.info("Uploaded files cleared. Using default knowledge base.")

    # Status Banner
    if st.session_state.active_store is not None and st.session_state.active_docs:
        docs_str = ", ".join([f"`{d}`" for d in st.session_state.active_docs])
        st.markdown(
            f'<div class="status-badge status-uploaded">🟢 Custom Documents Active</div>',
            unsafe_allow_html=True
        )
        st.caption(f"**Indexed:** {docs_str} ({st.session_state.chunk_count} chunks)")
    else:
        st.markdown(
            f'<div class="status-badge status-default">📁 Default Knowledge Base Active</div>',
            unsafe_allow_html=True
        )
        st.caption("Using sample AI documents (7 files). Upload custom files above to query your own documents.")

    st.divider()

    st.subheader("⚙️ Controls")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear conversation messages"):
            st.session_state.messages = []
            st.rerun()

    with col_btn2:
        if st.button("🧹 Clear All", use_container_width=True, help="Clear cache, documents & reset all"):
            clear_everything()
            st.success("All caches & documents cleared!")
            st.rerun()

    st.divider()
    st.caption("Supported formats: **PDF, DOCX, TXT**")
    st.caption("⚡ Powered by Google Gemini & LangChain FAISS")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown('<div class="main-title">IntelliAssist AI</div>', unsafe_allow_html=True)
if st.session_state.active_store is not None and st.session_state.active_docs:
    active_files_summary = ", ".join(st.session_state.active_docs)
    st.markdown(
        f'<div class="subtitle">Answering questions exclusively from your uploaded documents: <strong>{active_files_summary}</strong></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="subtitle">Ask questions about your documents with retrieval-augmented generation.</div>',
        unsafe_allow_html=True
    )


# ============================================================
# INITIAL GREETING IF EMPTY
# ============================================================

if not st.session_state.messages:
    with st.chat_message("assistant"):
        if st.session_state.active_store is not None and st.session_state.active_docs:
            docs_list = "\n".join([f"- 📄 **{doc}**" for doc in st.session_state.active_docs])
            st.markdown(
                f"**Welcome to IntelliAssist AI!**\n\n"
                f"I am ready to answer your questions based on your uploaded document(s):\n"
                f"{docs_list}\n\n"
                f"Ask me anything about their contents!"
            )
        else:
            st.markdown(
                "**Welcome to IntelliAssist AI.**\n\n"
                "I can answer questions using the information contained in your documents. "
                "Upload your own **PDF, DOCX, or TXT** files in the sidebar to start querying them directly, "
                "or ask questions about the pre-loaded AI knowledge base."
            )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                seen = set()
                for source in message["sources"]:
                    name = source.get("source", "Unknown")
                    page = source.get("page")
                    identifier = (name, page)
                    if identifier in seen:
                        continue
                    seen.add(identifier)

                    if page:
                        st.markdown(f"📄 **{name}** — Page {page}")
                    else:
                        st.markdown(f"📄 **{name}**")

        # Display retry button inside chat if it was an error message
        if message["role"] == "assistant" and message.get("is_error"):
            col_retry, _ = st.columns([1, 4])
            with col_retry:
                if st.button(f"🔄 Retry This Response", key=f"retry_msg_{idx}"):
                    # Find previous user prompt
                    if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                        st.session_state.retry_prompt = st.session_state.messages[idx - 1]["content"]
                        # Remove this failed assistant message
                        st.session_state.messages.pop(idx)
                        st.rerun()


# ============================================================
# REGENERATE / RETRY LAST RESPONSE BUTTON
# ============================================================

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    col_reg, col_space = st.columns([1, 4])
    with col_reg:
        if st.button("🔄 Regenerate / Retry Response", key="btn_regenerate_last"):
            # Find the user prompt for the last assistant response
            if len(st.session_state.messages) >= 2 and st.session_state.messages[-2]["role"] == "user":
                last_prompt = st.session_state.messages[-2]["content"]
                st.session_state.messages.pop()  # Remove last assistant response
                st.session_state.retry_prompt = last_prompt
                st.rerun()


# ============================================================
# CHAT INPUT & EXECUTION
# ============================================================

user_input = st.chat_input("Ask a question about your documents...")

# Determine prompt to run (new user input OR retry request)
prompt_to_run = None

if st.session_state.retry_prompt:
    prompt_to_run = st.session_state.retry_prompt
    st.session_state.retry_prompt = None
elif user_input:
    prompt_to_run = user_input
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    # Display immediately in chat
    with st.chat_message("user"):
        st.markdown(user_input)


# ============================================================
# EXECUTE RAG ANSWER GENERATION
# ============================================================

if prompt_to_run:
    # Select vector store strictly:
    # If custom files uploaded -> use active_store exclusively
    # If no custom files -> load base vector store
    if st.session_state.active_store is not None:
        target_store = st.session_state.active_store
    else:
        target_store = load_base_vector_store()

    with st.chat_message("assistant"):
        with st.spinner("Searching document context and generating answer..."):
            # Exclude current query if it's already at the end of messages for history context
            history_for_rag = [
                m for m in st.session_state.messages
                if m.get("content") != prompt_to_run
            ]

            result = generate_answer(
                prompt_to_run,
                chat_history=history_for_rag,
                store=target_store
            )

            answer = result["answer"]
            sources = result.get("sources", [])
            success = result.get("success", True)
            error_details = result.get("error")

            if not success:
                st.error("⚠️ AI Generation Error: High demand / service busy.")
                if error_details:
                    with st.expander("Details"):
                        st.caption(error_details)

            st.markdown(answer)

            if sources:
                with st.expander("📚 Sources"):
                    seen = set()
                    for source in sources:
                        name = source.get("source", "Unknown")
                        page = source.get("page")
                        identifier = (name, page)
                        if identifier in seen:
                            continue
                        seen.add(identifier)

                        if page:
                            st.markdown(f"📄 **{name}** — Page {page}")
                        else:
                            st.markdown(f"📄 **{name}**")

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "is_error": not success
    })

    st.rerun()