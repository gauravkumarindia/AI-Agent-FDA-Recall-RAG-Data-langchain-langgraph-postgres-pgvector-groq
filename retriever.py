# test_retriever.py
import getpass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# SUPABASE_PROJECT_REF = "YOUR_PROJECT_REF"
pwd = getpass.getpass("DB password: ")
CONN = f"postgresql://postgres:xxxxxxxxxxxxxxxxxxxxxxxx"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PGVector(embeddings=embeddings, collection_name="fda_recalls", connection=CONN, use_jsonb=True)

docs = vectorstore.similarity_search("blood pressure hypertension azilsartan Edarbi", k=10)
print(f"Found {len(docs)} docs")
for i, d in enumerate(docs):
    print(f"\n--- {i+1} ---\n{d.page_content[:500]}\nMETA: {d.metadata}\n")