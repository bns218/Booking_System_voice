"""
scripts/add_document.py
------------------------
CLI helper to add a new document to the RAG vector store without restarting the agent.

Usage:
    python scripts/add_document.py path/to/your/document.pdf
    python scripts/add_document.py docs/new_menu.pdf --preview
"""

import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DOCS_FOLDER
from rag import load_documents, build_vector_store


def main():
    parser = argparse.ArgumentParser(description="Add a document to the RAG knowledge base")
    parser.add_argument("filepath", help="Path to the document (.txt, .pdf, or .docx)")
    parser.add_argument("--preview", action="store_true", help="Preview extracted text without indexing")
    args = parser.parse_args()

    src = Path(args.filepath)
    if not src.exists():
        print(f"❌ File not found: {src}")
        sys.exit(1)

    allowed = {".txt", ".pdf", ".docx"}
    if src.suffix.lower() not in allowed:
        print(f"❌ Unsupported file type: {src.suffix}. Allowed: {', '.join(allowed)}")
        sys.exit(1)

    dest = DOCS_FOLDER / src.name
    DOCS_FOLDER.mkdir(exist_ok=True)

    if args.preview:
        from rag import LOADERS
        loader = LOADERS.get(src.suffix.lower())
        text = loader(src) if loader else ""
        print(f"\n── Preview: {src.name} ──────────────────────")
        print(text[:2000])
        if len(text) > 2000:
            print(f"\n... ({len(text) - 2000:,} more characters)")
        return

    # Copy to docs folder
    shutil.copy2(src, dest)
    print(f"📄 Copied {src.name} → {dest}")

    # Re-index
    print("🔢 Re-indexing knowledge base…")
    docs = load_documents(DOCS_FOLDER)
    build_vector_store(docs)
    print(f"✅ Done. '{src.name}' is now part of the RAG knowledge base.")


if __name__ == "__main__":
    main()
