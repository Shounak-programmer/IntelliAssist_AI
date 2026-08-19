import time
from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4

# Fallback model list if the primary model hits 503 high demand or quota limits
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
]
# Remove duplicates while preserving order
MODELS_TO_TRY = list(dict.fromkeys(FALLBACK_MODELS))


# ============================================================
# LOAD API KEY & CLIENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

_client = None
_embeddings = None
_vector_store = None


def get_gemini_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Check your .env file."
            )
        _client = genai.Client(
            api_key=api_key
        )
    return _client


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True
            }
        )
    return _embeddings


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        if (VECTOR_DB_DIR / "index.faiss").exists():
            _vector_store = FAISS.load_local(
                str(VECTOR_DB_DIR),
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
    return _vector_store


# ============================================================
# GEMINI API CALL WITH RETRY & FALLBACK
# ============================================================

def generate_content_with_retry(prompt: str, max_retries: int = 3, base_delay: float = 1.5):
    """
    Call Gemini API with automatic exponential backoff retry and fallback to alternative models
    in case of 503 UNAVAILABLE (high demand), 429 rate limits, or transient errors.
    """
    client = get_gemini_client()
    last_exception = None

    for model_name in MODELS_TO_TRY:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                if response and response.text:
                    return response.text.strip(), model_name
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                is_transient = (
                    "503" in err_str
                    or "unavailable" in err_str
                    or "high demand" in err_str
                    or "429" in err_str
                    or "resource_exhausted" in err_str
                    or "timeout" in err_str
                    or "deadline" in err_str
                )
                
                # If model is deprecated/not found (404), switch to next model immediately
                if "404" in err_str or "not_found" in err_str:
                    break

                if is_transient and attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
                elif is_transient:
                    # After exhausting retries on this model, break and try next fallback model
                    break
                else:
                    # For other non-transient errors, try next model or raise
                    break

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Failed to generate response after trying all available models.")


# ============================================================
# CONVERSATION HISTORY & QUERY CONTEXTUALIZATION
# ============================================================

def format_chat_history(chat_history: list = None) -> str:
    if not chat_history:
        return ""

    formatted = []
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def contextualize_query(query: str, chat_history: list = None) -> str:
    """If chat history exists, reformulate the query into a standalone search query."""
    if not chat_history:
        return query

    history_text = format_chat_history(chat_history)
    if not history_text:
        return query

    reformulation_prompt = f"""Given the following conversation history and a follow-up user question, rephrase the follow-up question into a concise, standalone question suitable for searching documents.
Resolve all pronouns (such as 'it', 'its', 'they', 'this', 'these') and context references using the conversation history.
Do NOT answer the question. Only return the reformulated question. If the question is already clear and standalone, return it as is.

CONVERSATION HISTORY:
{history_text}

FOLLOW-UP QUESTION:
{query}

STANDALONE QUESTION:"""

    try:
        reformulated, _ = generate_content_with_retry(reformulation_prompt, max_retries=2, base_delay=1.0)
        return reformulated if reformulated else query
    except Exception as e:
        print(f"Error reformulating query with history: {e}")
        return query


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query: str, store=None, k: int = TOP_K):
    target_store = store if store is not None else get_vector_store()

    if target_store is None:
        return []

    results = target_store.similarity_search(
        query,
        k=k
    )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get("page")

        if page:
            location = f"{source}, Page {page}"
        else:
            location = source

        context_parts.append(
            f"""SOURCE {i}
Location: {location}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query: str, chat_history: list = None, store=None, k: int = TOP_K):

    search_query = contextualize_query(query, chat_history)

    retrieved_documents = retrieve_documents(search_query, store=store, k=k)

    if not retrieved_documents:

        return {
            "answer": (
                "I could not find relevant information "
                "in the indexed documents."
            ),
            "sources": [],
            "success": True,
            "error": None
        }

    context = build_context(
        retrieved_documents
    )

    history_text = format_chat_history(chat_history)
    history_section = f"\nCONVERSATION HISTORY:\n{history_text}\n" if history_text else ""

    prompt = f"""You are IntelliAssist AI, a smart document question-answering assistant.

Answer the user's question using ONLY the provided document context while maintaining continuity with the ongoing conversation history.

Rules:

1. Use the retrieved documents as your source of truth.
2. Do not invent information.
3. If the answer is not contained in the documents, say that you could not find the information.
4. Give a clear and concise answer.
5. Maintain conversational context from previous messages when answering follow-up questions.
6. Do not fabricate citations.
{history_section}
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    try:
        answer, model_used = generate_content_with_retry(prompt, max_retries=3, base_delay=1.5)
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        answer = (
            "⚠️ The AI service is currently experiencing high demand. "
            "Please use the **Retry** button below to try again."
        )

    sources = []

    for document in retrieved_documents:

        sources.append({
            "source": document.metadata.get(
                "source",
                "Unknown source"
            ),
            "page": document.metadata.get("page")
        })

    return {
        "answer": answer,
        "sources": sources,
        "success": success,
        "error": error_msg
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("INTELLIASSIST AI — RAG TEST")
    print("=" * 70)

    question = input(
        "\nAsk a question about your documents: "
    ).strip()

    if not question:
        print("No question provided.")
        raise SystemExit

    print("\nSearching documents...")

    result = generate_answer(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    seen = set()

    for source in result["sources"]:

        name = source["source"]
        page = source["page"]

        identifier = (name, page)

        if identifier in seen:
            continue

        seen.add(identifier)

        if page:
            print(f"📄 {name} — Page {page}")
        else:
            print(f"📄 {name}")

    print("\n" + "=" * 70)