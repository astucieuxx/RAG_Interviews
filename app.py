"""
app.py — Streamlit RAG chat for competitive intelligence interviews

Usage:
    streamlit run app.py

Requires OPENAI_API_KEY environment variable.
Run ingest.py first to populate the vector database.
"""

import os
import sys
from pathlib import Path

import streamlit as st
import chromadb
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────
DB_DIR = Path("./chromadb_data")
COLLECTION_NAME = "interview_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # cost-effective for RAG; swap to gpt-4o if needed
TOP_K = 8  # number of chunks to retrieve

VENDORS = ["decagon", "sierra", "intercom", "forethought"]

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="CI Interview Analysis",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def init_clients():
    """Initialize OpenAI client and ChromaDB connection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ Set OPENAI_API_KEY environment variable")
        st.stop()

    if not DB_DIR.exists():
        st.error("❌ Database not found. Run `python ingest.py` first.")
        st.stop()

    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        st.error(f"❌ Collection '{COLLECTION_NAME}' not found. Run `python ingest.py` first.")
        st.stop()

    return client, collection


def get_query_embedding(client: OpenAI, query: str) -> list[float]:
    """Generate embedding for a search query."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def search_chunks(collection, query_vector: list[float], vendor_filter: list[str] | None = None,
                  source_filter: list[str] | None = None, top_k: int = TOP_K) -> list[dict]:
    """Search ChromaDB for relevant chunks with optional filters."""
    # Build where clause for filters
    where_clause = {}
    if vendor_filter and len(vendor_filter) < len(VENDORS):
        where_clause["vendor"] = {"$in": vendor_filter}
    if source_filter and len(source_filter) < 2:
        where_clause["source_type"] = {"$in": source_filter}
    
    # Query ChromaDB
    # Over-fetch if we have filters to apply
    query_k = top_k * 3 if (vendor_filter or source_filter) else top_k
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=query_k,
        where=where_clause if where_clause else None,
    )
    
    # Convert ChromaDB results to our format
    chunks = []
    if results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            chunk = {
                "text": results["documents"][0][i],
                "vendor": results["metadatas"][0][i]["vendor"],
                "source_type": results["metadatas"][0][i]["source_type"],
                "filename": results["metadatas"][0][i]["filename"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
            }
            chunks.append(chunk)
    
    # Apply additional filtering if needed (in case where clause didn't work perfectly)
    if vendor_filter and len(vendor_filter) < len(VENDORS):
        chunks = [c for c in chunks if c["vendor"] in vendor_filter]
    if source_filter and len(source_filter) < 2:
        chunks = [c for c in chunks if c["source_type"] in source_filter]
    
    # Take top_k
    return chunks[:top_k]


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    if not chunks:
        return "No relevant interview excerpts found for this query."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[Excerpt {i} | Vendor: {chunk['vendor'].upper()} | "
            f"Source: {chunk['source_type']} | File: {chunk['filename']}]"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def get_chat_response(client: OpenAI, query: str, context: str,
                      chat_history: list[dict]) -> str:
    """Generate a response using GPT with RAG context."""
    system_prompt = """You are an expert competitive intelligence analyst helping analyze 
interview transcripts from ex-customers and ex-employees of AI customer service vendors 
(Decagon, Sierra, Intercom/Fin, and Forethought).

Your role:
- Answer questions based ONLY on the provided interview excerpts
- Cite specific sources (vendor, source type, filename) when making claims
- Highlight patterns, contradictions, and notable insights across interviews
- Be direct and analytical — this is for internal strategic decision-making
- If the excerpts don't contain enough information, say so clearly
- When comparing vendors, organize your response by vendor for clarity
- Respond in the same language the user writes in (Spanish or English)

IMPORTANT: Base your answers strictly on the provided context. Do not hallucinate 
or invent information not present in the excerpts."""

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (last 6 exchanges for context window management)
    for msg in chat_history[-12:]:
        messages.append(msg)

    # Add current query with context
    user_message = f"""Based on these interview excerpts:

{context}

---

Question: {query}"""

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
    )

    return response.choices[0].message.content


def main():
    client, collection = init_clients()

    # ── Header ─────────────────────────────────────────────────────
    st.title("🔍 CI Interview Analysis")
    st.caption("RAG-powered analysis of competitive intelligence interviews")

    # ── Sidebar: Filters ───────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        vendor_filter = st.multiselect(
            "Vendors",
            options=VENDORS,
            default=VENDORS,
            help="Filter retrieved context by vendor",
        )

        source_filter = st.multiselect(
            "Source Type",
            options=["ex-cliente", "ex-empleado"],
            default=["ex-cliente", "ex-empleado"],
            help="Filter by interview source type",
        )

        st.divider()

        top_k = st.slider(
            "Chunks to retrieve",
            min_value=3,
            max_value=20,
            value=TOP_K,
            help="Number of relevant excerpts to include as context",
        )

        st.divider()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption("💡 **Example queries:**")
        st.caption("• What are common complaints about Decagon's onboarding?")
        st.caption("• Compare Sierra vs Intercom from the customer perspective")
        st.caption("• What do ex-employees say about Forethought's AI accuracy?")
        st.caption("• ¿Cuáles son los buying factors más mencionados?")

    # ── Chat state ─────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.caption(
                            f"**{src['vendor'].upper()}** | {src['source_type']} | "
                            f"`{src['filename']}` | chunk {src['chunk_index']}"
                        )

    # ── Chat input ─────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about the interview transcripts..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching interviews..."):
                # Embed query
                query_vector = get_query_embedding(client, prompt)

                # Search with filters
                chunks = search_chunks(
                    collection,
                    query_vector,
                    vendor_filter=vendor_filter if vendor_filter else None,
                    source_filter=source_filter if source_filter else None,
                    top_k=top_k,
                )

                # Build context and get response
                context = build_context(chunks)
                response = get_chat_response(
                    client, prompt, context, st.session_state.chat_history
                )

            st.markdown(response)

            # Show sources
            if chunks:
                sources = [
                    {
                        "vendor": c["vendor"],
                        "source_type": c["source_type"],
                        "filename": c["filename"],
                        "chunk_index": c["chunk_index"],
                    }
                    for c in chunks
                ]
                with st.expander("📎 Sources"):
                    for src in sources:
                        st.caption(
                            f"**{src['vendor'].upper()}** | {src['source_type']} | "
                            f"`{src['filename']}` | chunk {src['chunk_index']}"
                        )
            else:
                sources = []

        # Update state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
        })
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
