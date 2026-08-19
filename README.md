# IntelliAssist AI 🤖

> **A Retrieval-Augmented Generation (RAG) Document Assistant** powered by **Google Gemini** and **LangChain FAISS**.

---

## 📌 Problem

Students, researchers, and professionals frequently struggle to quickly search, synthesize, and extract accurate answers from lengthy, fragmented information distributed across multiple documents (PDFs, DOCX, TXT). Manual searching is time-consuming, prone to oversight, and standard keyword search lacks deep semantic understanding.

---

## 💡 Solution

**IntelliAssist AI** is an end-to-end Retrieval-Augmented Generation (RAG) system that:
1. Ingests and processes custom documents in multiple formats.
2. Generates semantic embeddings to index document chunks into a vector database.
3. Retrieves the most relevant contextual passages using similarity search.
4. Synthesizes factual, grounded answers using Google Gemini while strictly citing source documents and page numbers.

---

## 🏗️ Architecture

```text
Documents (PDF / DOCX / TXT)
    ↓
Document Loader
    ↓
Text Preprocessing & Cleaning
    ↓
Recursive Character Chunking
    ↓
HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
FAISS Vector Store
    ↓
Similarity Search (Top-K Context Retrieval)
    ↓
Retrieved Document Context + Conversation History
    ↓
Google Gemini (LLM Generation with Fallback & Retry)
    ↓
Grounded Answer + Source Citations
```

---

## 🛠️ Technologies Used

- **Language & UI**: Python, [Streamlit](https://streamlit.io/)
- **Framework & Orchestration**: [LangChain](https://www.langchain.com/) (langchain-core, langchain-community, langchain-text-splitters)
- **Vector Database**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Embeddings**: [HuggingFace Embeddings](https://huggingface.co/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Large Language Model (LLM)**: [Google Gemini API](https://ai.google.dev/) (`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-flash-latest`)
- **Document Parsers**: [pypdf](https://pypi.org/project/pypdf/) (PDF), [python-docx](https://python-docx.readthedocs.io/) (DOCX)
- **Environment Management**: `python-dotenv`

---

## ✨ Features

- 📁 **Multiple Document Formats**: Supports PDF, DOCX, and TXT files with clean text parsing and page-level metadata tracking.
- 🔍 **Semantic Search**: Fast and accurate vector similarity retrieval using FAISS and HuggingFace MiniLM embeddings.
- 🧠 **RAG-Based Answers**: Answers are synthesized directly from retrieved context, minimizing hallucinations.
- 📌 **Source Attribution**: Transparent citations showing exact document names and page numbers for every generated response.
- 📤 **Dynamic Document Upload**: Upload custom documents via the sidebar to instantly create isolated vector stores for the uploaded files.
- 🛡️ **Grounded Responses & Out-of-Context Handling**: Clear rejection and graceful handling when information is not present in the indexed documents.
- 🔄 **Auto-Retry & Model Fallback**: Built-in exponential backoff retry mechanism with multi-model fallback to handle high-demand API spikes (503 / 429 errors).
- 🔁 **Regenerate & Retry Controls**: One-click regeneration and in-chat retry options.
- 🧹 **Cache & Reset Management**: Complete cache and artifact clearing controls to ensure clean document isolation across runs.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Shounak-programmer/IntelliAssist_AI.git
cd IntelliAssist_AI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (or copy from `.env.example`):
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```
> Obtain your API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the Application
```bash
streamlit run src/app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📂 Project Structure

```text
IntelliAssist_AI/
│
├── Documents/               # Default sample documents
├── uploaded_documents/      # Directory for user-uploaded documents (auto-cleaned)
├── src/
│   ├── __init__.py
│   ├── app.py               # Streamlit web application & UI
│   ├── document_loader.py   # Multi-format document parser (PDF, DOCX, TXT)
│   ├── rag_pipeline.py      # Retrieval-augmented generation pipeline & Gemini client
│   └── vector_store.py      # FAISS indexing and vector store management
│
├── .env.example             # Example environment variables template
├── .gitignore               # Ignored files (secrets, venv, vector db caches)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📜 License

This project is licensed under the MIT License.
