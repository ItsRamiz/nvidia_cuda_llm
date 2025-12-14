from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(__file__)  # path to vector_db/
PATH = os.path.join(BASE_DIR, "docs")  # vector_db/docs

CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/vectordb"
)

COLLECTION_NAME = "my_rag_collection"

EMBEDDING_MODEL = "text-embedding-3-small"  # or any other model


# ==========================================
# LOAD DOCUMENTS
# ==========================================

def load_documents():
    loader = DirectoryLoader(PATH, glob="*.txt")
    documents = loader.load()
    return documents


# ==========================================
# SPLIT DOCUMENTS
# ==========================================

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    return splitter.split_documents(documents)


# ==========================================
# EMBED + INSERT INTO PGVECTOR
# ==========================================

def store_in_pgvector(texts):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print(f"[+] Storing {len(texts)} chunks in PGVector...")

    PGVector.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

    print("[✓] Done! Data successfully saved into PGVector.")


# ==========================================
# MAIN
# ==========================================

def main():
    print("[+] Loading documents...")
    docs = load_documents()

    print("[+] Splitting documents...")
    chunks = split_text(docs)

    print("[+] Embedding and saving into PGVector...")
    store_in_pgvector(chunks)

if __name__ == "__main__":
    main()
