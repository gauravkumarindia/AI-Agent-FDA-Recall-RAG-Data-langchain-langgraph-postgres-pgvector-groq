# Project by Gaurav - openFDA RAG ETL - 2026
# Stack: LangChain + HuggingFace (384-dim) + Supabase Postgres pgvector
# Run: python 1_fetch_and_embed.py

import requests
from getpass import getpass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# 1. Config - keep keys out of GitHub
SUPABASE_PASSWORD = getpass("Enter Supabase DB password: ")

# Use Direct connection on laptop (not pooler). Pooler was only for Colab IPv6 fix.
CONNECTION_STRING = f"postgresql://postgres:xxxxxxxxxxxxxxxxxxxxx"

API_URL = "https://api.fda.gov/drug/enforcement.json"
PARAMS = {"limit": 20} # keep small for slow laptop, increase later

def fetch_recalls():
    print("Fetching from openFDA...")
    r = requests.get(API_URL, params=PARAMS, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    print(f"Got {len(results)} recalls")
    return results

def to_documents(results):
    docs = []
    for item in results:
        text = (
            f"Recall {item.get('recall_number', 'N/A')} by {item.get('recalling_firm', 'N/A')}. "
            f"Product: {item.get('product_description', 'N/A')}. "
            f"Reason for recall: {item.get('reason_for_recall', 'N/A')}. "
            f"Classification: {item.get('classification', 'N/A')}. "
            f"Status: {item.get('status', 'N/A')}."
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "recall_number": item.get("recall_number"),
                "firm": item.get("recalling_firm"),
                "source": "openFDA"
            }
        ))
    return docs

def main():
    # 2. Fetch and convert
    results = fetch_recalls()
    docs = to_documents(results)

    # 3. Chunk for RAG - small chunks work better for recall reasons
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 4. Embeddings - 384-dim, same as your employee project, fast on laptop
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Push to Postgres pgvector
    print("Inserting to Supabase pgvector...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="fda_recalls", # new table, separate from langchain_pg_embedding
        connection=CONNECTION_STRING,
        pre_delete_collection=True, # clean rerun
    )
    print("Done. Check Supabase table: fda_recalls")

if __name__ == "__main__":
    main()