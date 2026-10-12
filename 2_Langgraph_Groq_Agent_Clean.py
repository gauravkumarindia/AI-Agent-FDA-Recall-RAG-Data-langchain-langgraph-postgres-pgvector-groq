import os
import getpass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# --- 1. Get keys ---
groq_key = os.getenv("GROQ_API_KEY") or getpass.getpass("Enter Groq API key: ")
supabase_pass = getpass.getpass("Enter Supabase DB password: ")

# --- IMPORTANT: Put your PROJECT_REF here ---
SUPABASE_PROJECT_REF = ""  # e.g. abcdefghijklmno
CONNECTION_STRING = f"postgresql://postgres:xxxxxxxxxxxxxxxxxxxxxx"

# --- 2. Embeddings (same as 1_fetch_and_embed.py) ---
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 3. VectorStore - ONLY fda_recalls collection ---
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="fda_recalls",
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# --- 4. Retriever that is STRICTLY filtered to fda_recalls ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # get 5 chunks
)

@tool
def search_fda_recalls(query: str) -> str:
    """Search FDA drug recalls stored in Supabase pgvector. Use this for any question about recalls, firms, reasons, recall numbers."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant FDA recall found in database."
    result = []
    for d in docs:
        meta = d.metadata
        firm = meta.get("firm", "Unknown firm")
        result.append(f"Firm: {firm}\nContent: {d.page_content}\nMetadata: {meta}")
    return "\n\n---\n\n".join(result)

# --- 5. LLM - Groq with openai/gpt-oss-120b (works on Groq) ---
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0,  # 0 = no hallucination
)

# --- 6. Strict RAG system prompt ---
SYSTEM_PROMPT = """You are a FDA Recall Assistant.

RULES:
1. Answer ONLY from the tool output (search_fda_recalls).
2. If tool returns no relevant info, say: "Not found in my 20 FDA records in Supabase. I only have 20 recent recalls."
3. Do NOT use your own training knowledge about drugs.
4. Always cite Recall Number and Firm from context.
5. If question is about blood pressure drugs but context has none, clearly say none found in your database.
"""

# --- 7. Create agent (new API, no deprecation) ---
try:
    from langchain.agents import create_agent
    agent = create_agent(
        model=llm,
        tools=[search_fda_recalls],
        system_prompt=SYSTEM_PROMPT,
    )
    USE_NEW_API = True
except ImportError:
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    agent = create_react_agent(llm, [search_fda_recalls], prompt=SYSTEM_PROMPT, checkpointer=memory)
    USE_NEW_API = False
    print("Using fallback langgraph prebuilt agent")

print("\nFDA Agent ready with openai/gpt-oss-120b. Ask about recalls (type exit to quit)\n")

config = {"configurable": {"thread_id": "fda-thread-1"}}

while True:
    q = input("You: ").strip()
    if q.lower() in ["exit", "quit", "bye"]:
        break
    if not q:
        continue
    
    try:
        if USE_NEW_API:
            result = agent.invoke({"messages": [{"role": "user", "content": q}]})
            # last AI message
            last = result["messages"][-1]
            print(f"\nAgent: {last.content}\n")
        else:
            result = agent.invoke({"messages": [("user", q)]}, config=config)
            last = result["messages"][-1]
            print(f"\nAgent: {last.content}\n")
    except Exception as e:
        print(f"\nError: {e}\n")
