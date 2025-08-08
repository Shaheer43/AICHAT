# reindex.py
from rag_chain import load_documents, create_faiss_index

def reindex():
    print("🚀 Starting reindexing...")
    docs = load_documents("Source")
    create_faiss_index(docs)
    print("✅ Reindex complete.")

if __name__ == "__main__":
    reindex()
