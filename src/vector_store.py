import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.document_loader import load_all_documents


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "Documents"
VECTOR_DB_DIR = Path(__file__).parent.parent / "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --------------------------------------------------
# BUILD AND SAVE VECTOR STORE
# --------------------------------------------------

def build_and_save_vector_store():
    # STEP 1: LOAD DOCUMENTS
    print("\nLoading documents...")
    raw_documents = load_all_documents(DATA_DIR)
    print(f"Loaded {len(raw_documents)} document pages/files.")

    # STEP 2: CONVERT TO LANGCHAIN DOCUMENTS
    documents = []
    for item in raw_documents:
        documents.append(
            Document(
                page_content=item["text"],
                metadata=item["metadata"]
            )
        )
    print(f"Converted {len(documents)} documents to LangChain format.")

    # STEP 3: SPLIT INTO CHUNKS
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # STEP 4: LOAD EMBEDDING MODEL
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    print("Embedding model loaded.")

    # STEP 5: CREATE FAISS VECTOR STORE
    print("\nGenerating embeddings and building FAISS index...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print("FAISS index created.")

    # STEP 6: SAVE VECTOR STORE
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_DB_DIR))
    print(f"\nVector database saved to: {VECTOR_DB_DIR}")

    return vector_store


if __name__ == "__main__":
    vector_store = build_and_save_vector_store()

    # STEP 7: TEST SEMANTIC SEARCH
    query = "What is Retrieval-Augmented Generation?"

    print(f"\nTesting semantic search:")
    print(f"Query: {query}")

    results = vector_store.similarity_search(
        query,
        k=3
    )

    print("\nTop results:\n")

    for i, result in enumerate(results, start=1):

        print("=" * 70)

        print(f"Result {i}")

        print(f"Source: {result.metadata.get('source')}")

        print(f"Page: {result.metadata.get('page')}")

        print(f"\n{result.page_content[:500]}")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)