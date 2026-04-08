"""
app.py — Streamlit RAG chat for competitive intelligence interviews

Usage:
    streamlit run app.py

Requires OPENAI_API_KEY environment variable.
Run ingest.py first to populate the vector database.
"""

import os
import sys
import re
from collections import defaultdict
from pathlib import Path

import streamlit as st
import pickle
import numpy as np
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────
DB_DIR = Path("./vector_db")
DB_FILE = DB_DIR / "embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # cost-effective for RAG; swap to gpt-4o if needed
TOP_K = 8  # number of chunks to retrieve

VENDORS = ["decagon", "sierra", "intercom", "forethought"]
SOURCE_TYPES = ["ex-customer", "ex-employee"]


def normalize_source_type(source_type: str) -> str:
    """Normalize legacy Spanish tags to English canonical labels."""
    mapping = {
        "ex-cliente": "ex-customer",
        "ex-empleado": "ex-employee",
    }
    return mapping.get(source_type, source_type)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG AI Agents Competitors Analysis",
    page_icon="🔎",
    layout="wide",
)


def inject_global_styles():
    st.markdown(
        """
<style>
/* Zendesk-inspired dark palette */
:root{
  --zd-bg: #031A1D;         /* deep teal */
  --zd-panel: #062A30;      /* slightly lighter teal */
  --zd-border: rgba(255,255,255,0.18);
  --zd-text: #F4F8F9;
  --zd-subtle: #D7E4E7;
  --zd-muted: #B7C8CC;
  --zd-teal: #03363D;       /* Zendesk evergreen */
  --zd-green: #78A300;      /* Zendesk green */
  --zd-cyan: #2F9DB5;       /* accent */
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--zd-bg) !important;
  color: var(--zd-text);
}
* { color: var(--zd-text); }

/* Streamlit top header/menu bar */
header[data-testid="stHeader"] {
  background: rgba(3,26,29,0.95) !important;
  border-bottom: 1px solid var(--zd-border) !important;
}
div[data-testid="stToolbar"] {
  background: transparent !important;
}

/* Layout + typography */
.main .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1220px; }
h1, h2, h3 { letter-spacing: -0.02em; }
div[data-testid="stCaptionContainer"] p { color: var(--zd-subtle); }

/* Header card */
.ci-header {
  border: 1px solid var(--zd-border);
  background: linear-gradient(135deg, rgba(3,54,61,0.85), rgba(6,42,48,0.85));
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 12px;
}
.ci-header-title { font-size: 1.35rem; font-weight: 700; margin: 0; }
.ci-header-sub { margin: 6px 0 0 0; color: var(--zd-subtle); font-size: 0.95rem; }
.ci-header-meta {
  margin-top: 10px;
  font-size: 0.86rem;
  color: var(--zd-muted);
}
.ci-header-owner {
  margin-top: 6px;
  font-size: 0.82rem;
  color: var(--zd-muted);
}

/* “Badge” chips */
.chip {
  display: inline-flex; align-items: center;
  border: 1px solid var(--zd-border);
  background: rgba(255,255,255,0.03);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.82rem;
  color: var(--zd-subtle);
  gap: 8px;
}
.dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
.dot-purple { background: var(--zd-green); }
.dot-blue { background: var(--zd-cyan); }
.dot-emerald { background: #22c55e; }

/* Chat message spacing */
div[data-testid="stChatMessage"] {
  border-radius: 14px;
  border: 1px solid var(--zd-border);
  background: rgba(255,255,255,0.02);
}
div[data-testid="stChatMessage"] p { color: var(--zd-text); }

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Sidebar background */
section[data-testid="stSidebar"] {
  background: rgba(6,42,48,0.82) !important;
  border-right: 1px solid var(--zd-border);
}

/* Metric cards */
div[data-testid="stMetric"]{
  border: 1px solid var(--zd-border);
  background: rgba(255,255,255,0.03);
  border-radius: 14px;
  padding: 10px 12px;
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: var(--zd-subtle) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--zd-text) !important;
}

/* Dataframe container */
div[data-testid="stDataFrame"]{
  border: 1px solid var(--zd-border);
  border-radius: 12px;
  overflow: hidden;
}

/* Inputs/selects/buttons contrast */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  background: #0A2F35 !important;
  border-color: var(--zd-border) !important;
}
div[data-baseweb="select"] *, div[data-baseweb="input"] *, div[data-baseweb="textarea"] * {
  color: var(--zd-text) !important;
}

/* Chat input container + field */
div[data-testid="stChatInput"] {
  background: #0A2F35 !important;
  border: 1px solid var(--zd-border) !important;
  border-radius: 12px !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.22);
  overflow: hidden !important;
}
/* Streamlit bottom/floating chat bar wrappers (version-dependent selectors) */
div[data-testid="stBottomBlockContainer"],
div[data-testid="stChatFloatingInputContainer"],
div[data-testid="stChatInputContainer"],
.stChatFloatingInputContainer {
  background: var(--zd-bg) !important;
  border-top: 1px solid rgba(255,255,255,0.08) !important;
}
div[data-testid="stBottomBlockContainer"] > div,
div[data-testid="stChatFloatingInputContainer"] > div,
div[data-testid="stChatInputContainer"] > div,
.stChatFloatingInputContainer > div {
  background: var(--zd-bg) !important;
  padding-top: 6px !important;
  padding-bottom: 6px !important;
}
/* Remove white "card" wrappers used by some Streamlit builds */
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] > div > div {
  background: transparent !important;
}
div[data-testid="stChatInput"] textarea {
  color: var(--zd-text) !important;
  background: transparent !important;
  caret-color: var(--zd-text) !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
  color: var(--zd-muted) !important;
}
div[data-testid="stChatInput"] button {
  background: transparent !important;
  border: none !important;
}
div[data-testid="stChatInput"] svg {
  fill: var(--zd-subtle) !important;
}

button[kind], .stButton > button {
  background: #0A2F35 !important;
  color: var(--zd-text) !important;
  border: 1px solid var(--zd-border) !important;
}
button[kind]:hover, .stButton > button:hover {
  background: #11424A !important;
  border-color: rgba(255,255,255,0.28) !important;
}

/* Expanders and helper text */
details, div[data-testid="stExpander"] {
  border-color: var(--zd-border) !important;
}
small, .stMarkdown, .stCaption { color: var(--zd-subtle); }

/* Context callout */
.context-callout{
  border: 1px solid var(--zd-border);
  background: rgba(255,255,255,0.025);
  border-radius: 12px;
  padding: 10px 12px;
  margin: 6px 0 14px 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def init_clients():
    """Initialize OpenAI client and load vector database."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ Set OPENAI_API_KEY environment variable")
        st.stop()

    if not DB_FILE.exists():
        st.error("❌ Database not found. Run `python ingest.py` first.")
        st.stop()

    client = OpenAI(api_key=api_key)
    
    # Load database
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)

    return client, database


def get_query_embedding(client: OpenAI, query: str) -> list[float]:
    """Generate embedding for a search query."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def is_count_interviews_query(query: str) -> bool:
    """Detect questions asking for total interview counts."""
    q = query.lower()
    patterns = [
        r"\bhow many interviews\b",
        r"\bnumber of interviews\b",
        r"\btotal interviews\b",
        r"\bcu[aá]ntas entrevistas\b",
        r"\bcu[aá]ntos interviews\b",
        r"\btotal de entrevistas\b",
    ]
    return any(re.search(p, q) for p in patterns)


def count_interviews(database, vendor_filter: list[str] | None = None,
                     source_filter: list[str] | None = None) -> tuple[int, dict]:
    """Count unique interview files in database after applying filters."""
    valid_vendors = set(vendor_filter) if vendor_filter else set(VENDORS)
    valid_sources = set(source_filter) if source_filter else set(SOURCE_TYPES)

    per_file = {}
    for chunk in database["chunks"]:
        vendor = chunk["vendor"]
        source_type = normalize_source_type(chunk["source_type"])
        filename = chunk["filename"]
        if vendor not in valid_vendors or source_type not in valid_sources:
            continue
        # One interview per file
        per_file[filename] = (vendor, source_type)

    breakdown = {}
    for vendor, source_type in per_file.values():
        key = f"{vendor} ({source_type})"
        breakdown[key] = breakdown.get(key, 0) + 1
    return len(per_file), dict(sorted(breakdown.items()))


def get_database_interview_summary(database) -> tuple[int, list[dict]]:
    """Return total interviews and per vendor/source interview counts."""
    interviews = {}
    for chunk in database["chunks"]:
        filename = chunk["filename"]
        interviews[filename] = (
            chunk["vendor"],
            normalize_source_type(chunk["source_type"]),
        )

    summary = defaultdict(lambda: {"ex-customer": 0, "ex-employee": 0})
    for vendor, source_type in interviews.values():
        if source_type in SOURCE_TYPES:
            summary[vendor][source_type] += 1

    rows = []
    for vendor in VENDORS:
        rows.append({
            "company": vendor,
            "customers": summary[vendor]["ex-customer"],
            "former employees": summary[vendor]["ex-employee"],
            "total": summary[vendor]["ex-customer"] + summary[vendor]["ex-employee"],
        })
    return len(interviews), rows


def get_database_chunk_count(database) -> int:
    return int(len(database.get("chunks", [])))


def format_count_response(query: str, total: int, breakdown: dict) -> str:
    """Format count response in user language (Spanish/English)."""
    is_spanish = any(token in query.lower() for token in ["cuánt", "cuant", "entrevistas", "cuantos", "cuantas"])
    if is_spanish:
        lines = [f"Hay **{total} entrevistas** en los filtros actuales."]
        if breakdown:
            lines.append("")
            lines.append("Desglose:")
            lines.extend([f"- {k}: {v}" for k, v in breakdown.items()])
        return "\n".join(lines)

    lines = [f"There are **{total} interviews** with the current filters."]
    if breakdown:
        lines.append("")
        lines.append("Breakdown:")
        lines.extend([f"- {k}: {v}" for k, v in breakdown.items()])
    return "\n".join(lines)


def search_chunks(database, query_vector: list[float], vendor_filter: list[str] | None = None,
                  source_filter: list[str] | None = None, top_k: int = TOP_K) -> list[dict]:
    """Search vector database for relevant chunks with optional filters."""
    embeddings = database["embeddings"]
    chunks_data = database["chunks"]
    
    # Convert query to numpy array
    query_vec = np.array([query_vector], dtype=np.float32)
    
    # Calculate cosine similarity (1 - cosine distance)
    # Use dot product for efficiency (normalized vectors)
    similarities = np.dot(embeddings, query_vec.T).flatten()
    
    # Get top indices
    query_k = top_k * 3 if (vendor_filter or source_filter) else top_k
    top_indices = np.argsort(similarities)[::-1][:query_k]
    
    # Get chunks and apply filters
    results = []
    for idx in top_indices:
        chunk = {
            "text": chunks_data[idx]["text"],
            "vendor": chunks_data[idx]["vendor"],
            "source_type": normalize_source_type(chunks_data[idx]["source_type"]),
            "filename": chunks_data[idx]["filename"],
            "chunk_index": chunks_data[idx]["chunk_index"],
            "similarity": float(similarities[idx]),
        }
        
        # Apply filters
        if vendor_filter and len(vendor_filter) < len(VENDORS):
            if chunk["vendor"] not in vendor_filter:
                continue
        if source_filter and len(source_filter) < len(SOURCE_TYPES):
            if chunk["source_type"] not in source_filter:
                continue
        
        results.append(chunk)
    
    # Sort by similarity and take top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


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
    inject_global_styles()
    client, database = init_clients()
    db_total_interviews, db_summary_rows = get_database_interview_summary(database)
    db_total_chunks = get_database_chunk_count(database)

    # ── Header ─────────────────────────────────────────────────────
    st.markdown(
        """
<div class="ci-header">
  <div class="ci-header-title">RAG AI Agents Competitors Analysis</div>
  <div class="ci-header-sub">
    Competitive intelligence chat over ex-customer & ex-employee interviews. Answers are grounded in retrieved excerpts.
  </div>
  <div class="ci-header-meta">
    Grounded in 40+ structured interviews across customers and former employees from Decagon, Sierra, Intercom, Forethought, and Zendesk.
  </div>
  <div class="ci-header-owner">
    Market Intelligence initiative. Contacts: Benjamin Miranda and Rodrigo Gomez.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3 = st.columns([1, 1, 2])
    with kpi1:
        st.metric("Interviews indexed", db_total_interviews)
    with kpi2:
        st.metric("Chunks indexed", db_total_chunks)
    with kpi3:
        st.markdown(
            """
<span class="chip"><span class="dot dot-purple"></span>RAG: vector search + citations</span>
&nbsp;&nbsp;
<span class="chip"><span class="dot dot-blue"></span>Filters: vendor + source type</span>
&nbsp;&nbsp;
<span class="chip"><span class="dot dot-emerald"></span>Language: matches your prompt</span>
            """,
            unsafe_allow_html=True,
        )

    # ── Sidebar: Filters ───────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        vendor_filter = st.multiselect(
            "Vendors",
            options=VENDORS,
            default=VENDORS,
            help="Filter retrieved context by vendor",
        )

        source_filter = st.multiselect(
            "Source Type",
            options=SOURCE_TYPES,
            default=SOURCE_TYPES,
            help="Filter by interview source type",
        )

        st.divider()
        st.subheader("Database coverage")
        st.caption(f"Indexed interviews: **{db_total_interviews}** · Chunks: **{db_total_chunks}**")
        st.dataframe(
            db_summary_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        with st.expander("Methodology & scope", expanded=False):
            st.markdown(
                """
- **Evidence base:** 40+ structured interviews across enterprise customers and former employees.
- **Vendors covered:** Decagon, Sierra, Intercom, Forethought, and Zendesk context interviews.
- **Lens:** buyer journey from evaluation to implementation and optimization.
- **Research intent:** directional, interview-backed market intelligence (not a definitive market report).
- **Best use:** compare patterns, validate hypotheses, and identify follow-up questions.
                """
            )

        st.divider()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        with st.expander("Examples", expanded=False):
            st.caption("• What are common complaints about Decagon's onboarding?")
            st.caption("• Compare Sierra vs Intercom from the customer perspective")
            st.caption("• What do ex-employees say about Forethought's AI accuracy?")
            st.caption("• ¿Cuáles son los buying factors más mencionados?")

    # ── Chat state ─────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown(
        """
<div class="context-callout">
<strong>How to interpret answers</strong><br/>
Responses are generated from retrieved interview excerpts, not from the full corpus at once.
Use vendor/source filters to narrow the lens, and ask follow-up questions to validate conclusions.
If evidence is insufficient, the assistant should explicitly say so.
</div>
        """,
        unsafe_allow_html=True,
    )

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
    if prompt := st.chat_input("Ask about the interviews… (e.g. 'Compare Sierra vs Intercom')"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching interviews..."):
                if is_count_interviews_query(prompt):
                    total, breakdown = count_interviews(
                        database,
                        vendor_filter=vendor_filter if vendor_filter else None,
                        source_filter=source_filter if source_filter else None,
                    )
                    response = format_count_response(prompt, total, breakdown)
                    chunks = []
                else:
                    # Embed query
                    query_vector = get_query_embedding(client, prompt)

                    # Search with filters
                    chunks = search_chunks(
                        database,
                        query_vector,
                        vendor_filter=vendor_filter if vendor_filter else None,
                        source_filter=source_filter if source_filter else None,
                        top_k=TOP_K,
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
