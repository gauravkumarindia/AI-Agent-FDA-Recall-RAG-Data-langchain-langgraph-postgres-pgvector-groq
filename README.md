# FDA Recall RAG Agent - LangGraph + Groq + Supabase pgvector

A small but complete **RAG (Retrieval-Augmented Generation)** pipeline built for portfolio:

- Fetches real FDA drug recall data from **openFDA API**
- Chunks and embeds with **sentence-transformers/all-MiniLM-L6-v2**
- Stores in **Supabase Postgres pgvector** (`langchain_pg_collection` + `langchain_pg_embedding`)
- Answers questions via **LangGraph ReAct Agent + Groq** (`openai/gpt-oss-120b`)
- Strict RAG: answers ONLY from retrieved context, no hallucination

### ✅ Verified DB Content (20 recalls → 25 chunks)

```
Found 10 docs for "blood pressure hypertension"

1. Recall D-0429-2021 by Cardinal Health Inc.
   Product: Edarbi (azilsartan medoxomil) tablets 80 mg 30 Tablets
   Reason: CGMP Deviations: Intermittent exposure to temperature excursion
   Classification: Class II, Status: Terminated
   → This is a blood-pressure ARB drug

2. Recall D-518-2014 by Aidapak Services, LLC
   Product: FAMOTIDINE may be mislabeled as CARVEDILOL PHOSPHATE ER 20mg
   → Carvedilol is a beta-blocker (blood pressure)
```

So for the question **"What blood pressure drugs were recalled?"** the correct RAG answer from this 20-record DB is:
**Edarbi (azilsartan medoxomil) D-0429-2021 - Cardinal Health Inc.**

### 🏗️ Architecture

```
openFDA API (limit=20) 
   → chunk (500 chars) 
   → HuggingFace Embeddings 
   → PGVector (Supabase)
   → Retriever (k=5) 
   → Groq Tool (search_fda_recalls)
   → LangGraph Agent (openai/gpt-oss-120b, temp=0)
```

### 📁 Files

- `1_fetch_and_embed.py` - Fetch + embed + store to Supabase
- `2_langgraph_groq_agent_clean.py` - Final RAG agent (strict prompt, collection filter)
- `retriever.py` - Debug script to test retriever directly
- `requirements.txt` - Dependencies

### 🚀 Setup

1. Create Supabase project, enable `vector` extension

2. keys
GROQ_API_KEY=gsk_...
SUPABASE_DB_PASSWORD=your_db_password
SUPABASE_PROJECT_REF=abcdefgh...
HF_TOKEN=optional_for_faster_download


3. Install
```bash
pip install -r requirements.txt
```

4. Fetch and embed (run once)
```bash
python 1_fetch_and_embed.py
# Creates collection fda_recalls with ~25 chunks
```

5. Test retriever (optional)
```bash
python retriever.py
```

6. Run agent
```bash
python 2_langgraph_groq_agent_clean.py
```

Example chat:
```
You: What blood pressure drugs were recalled?
Agent: The only blood-pressure medication in the recent FDA recall records is:
- Edarbi (azilsartan medoxomil) 80 mg tablets – Recall Number D-0429-2021, 
  issued by Cardinal Health Inc. (Class II, Terminated).
```

### 🔒 Why this is true RAG

- System prompt: `Answer ONLY from tool output. If no relevant info, say Not found in my 20 FDA records.`
- Temperature = 0
- Collection filter = `fda_recalls` only (employees_demo deleted)
- Proven by retriever.py log above

### 🔧 To get more blood pressure data

In `1_fetch_and_embed.py` change params:
```python
params = {
  "limit": 300,
  "search": 'product_description:"blood pressure" OR hypertension OR losartan OR lisinopril'
}
```
Then re-run fetch.

### Author

Built with LangChain, LangGraph, Groq, Supabase pgvector.
