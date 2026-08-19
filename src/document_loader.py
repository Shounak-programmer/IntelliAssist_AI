from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    lines = []

    for line in text.split("\n"):
        line = " ".join(line.split())

        if line:
            lines.append(line)

    # Rebuild text with clean line separation
    text = "\n".join(lines)

    # Remove excessive blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def load_pdf(file_path: Path):
    """
    Extract text from a PDF while preserving page numbers.
    """

    reader = PdfReader(str(file_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        text = clean_text(text)

        if text:
            documents.append({
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "file_type": "pdf",
                    "page": page_number
                }
            })

    return documents


def load_txt(file_path: Path):
    """
    Extract text from a TXT file.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    text = clean_text(text)

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": "txt",
            "page": None
        }
    }]


def load_docx(file_path: Path):
    """
    Extract text from a DOCX file.
    """

    doc = DocxDocument(str(file_path))

    paragraphs = []

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    text = clean_text(text)

    if not text:
        return []

    return [{
        "text": text,
        "metadata": {
            "source": file_path.name,
            "file_type": "docx",
            "page": None
        }
    }]


def load_document(file_path: Path):
    """
    Automatically select the appropriate loader.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        print(f"Warning: Skipping empty or missing file {file_path.name}")
        return []

    extension = file_path.suffix.lower()

    try:
        if extension == ".pdf":
            return load_pdf(file_path)
        elif extension == ".txt":
            return load_txt(file_path)
        elif extension == ".docx":
            return load_docx(file_path)
        else:
            print(f"Warning: Unsupported file type: {extension}")
            return []
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return []


def load_all_documents(directory: str):
    """
    Load every supported document from a directory.
    """
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        return []

    all_documents = []

    supported_extensions = {
        ".pdf",
        ".txt",
        ".docx"
    }

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            if file_path.stat().st_size == 0:
                continue
            print(f"Loading: {file_path.name}")
            documents = load_document(file_path)
            all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "Documents"

    documents = load_all_documents(DATA_DIR)

    print("\n" + "=" * 60)
    print("DOCUMENT EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Total extracted documents/pages: {len(documents)}")

    for i, document in enumerate(documents, start=1):

        metadata = document["metadata"]

        print(
            f"\n[{i}] {metadata['source']} "
            f"| Page: {metadata['page']}"
        )

        print(
            f"Characters: {len(document['text'])}"
        )

        print(
            f"Preview: {document['text'][:200]}..."
        )