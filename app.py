# =========================================================
# app.py — Agentic RAG Pipeline: Sales & Lead Qualification
# Construction & Logistics SaaS — Streamlit Deployment
# =========================================================

import os
import json
import time
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LeadIntel — Logistics SaaS",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0d0f14; }

    .stApp {
        background: #0d0f14;
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111420 !important;
        border-right: 1px solid #1e2435;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 50%, #0f172a 100%);
        border: 1px solid #2dd4bf22;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .header-banner::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, #2dd4bf18 0%, transparent 70%);
        border-radius: 50%;
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 26px;
        font-weight: 600;
        color: #2dd4bf;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .header-sub {
        font-size: 14px;
        color: #64748b;
        margin: 0;
        font-weight: 300;
    }

    /* Cards */
    .metric-card {
        background: #111420;
        border: 1px solid #1e2435;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Agent status badges */
    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-research  { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb44; }
    .badge-rag       { background: #1e3a2f; color: #34d399; border: 1px solid #059669aa; }
    .badge-qualify   { background: #3a1e2f; color: #f472b6; border: 1px solid #db277744; }
    .badge-coord     { background: #2d1e3a; color: #c084fc; border: 1px solid #9333ea44; }

    /* Score display */
    .score-hot   { color: #f87171; font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 600; }
    .score-warm  { color: #fbbf24; font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 600; }
    .score-cold  { color: #60a5fa; font-family: 'IBM Plex Mono', monospace; font-size: 32px; font-weight: 600; }

    /* Output box */
    .output-box {
        background: #080b11;
        border: 1px solid #1e2435;
        border-radius: 10px;
        padding: 20px 24px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.7;
        max-height: 520px;
        overflow-y: auto;
        white-space: pre-wrap;
    }

    /* Step indicator */
    .step-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #1e243511;
    }
    .step-num {
        width: 26px; height: 26px;
        border-radius: 50%;
        background: #1e2435;
        border: 1px solid #2dd4bf44;
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #2dd4bf;
        flex-shrink: 0;
    }
    .step-text {
        font-size: 13px;
        color: #94a3b8;
    }

    /* Divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2dd4bf33, transparent);
        margin: 24px 0;
    }

    /* Streamlit overrides */
    .stTextArea textarea {
        background: #080b11 !important;
        border: 1px solid #1e2435 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 14px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2dd4bf, #0891b2) !important;
        color: #0d0f14 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
        padding: 10px 28px !important;
        letter-spacing: 0.05em !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px #2dd4bf33 !important;
    }
    .stFileUploader {
        border: 1px dashed #2dd4bf44 !important;
        border-radius: 8px !important;
        background: #080b11 !important;
    }
    label, .stSelectbox label, .stTextArea label {
        color: #64748b !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    .stSpinner > div {
        border-color: #2dd4bf transparent transparent transparent !important;
    }
    hr { border-color: #1e2435 !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# API KEY SETUP (from Streamlit secrets or sidebar input)
# =========================================================

def get_api_key() -> str:
    """
    Load API key from Streamlit secrets (production) or
    from the sidebar input field (demo/portfolio mode).
    Secrets take priority.
    """
    # Try st.secrets first (Streamlit Cloud deployment)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # Fall back to sidebar input
    return st.session_state.get("api_key_input", "")


# =========================================================
# CACHED RESOURCE: EMBEDDING MODEL
# Loading sentence-transformers is slow (~5s).
# @st.cache_resource ensures it loads once per session.
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# CACHED RESOURCE: VECTOR DATABASE
# Built once from uploaded PDFs. Rebuilt if PDFs change.
# =========================================================

@st.cache_resource
def build_vector_db(pdf_texts: tuple):
    """
    Build a ChromaDB collection from extracted PDF texts.

    Args:
        pdf_texts: tuple of (filename, text) pairs.
                   Must be a tuple (not list) for st.cache_resource
                   to hash correctly.

    Returns:
        chromadb collection
    """
    embedding_model = load_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chroma_client = chromadb.Client()

    # Use get_or_create to avoid duplicate collection errors
    # on re-runs within the same session
    try:
        chroma_client.delete_collection("logistics_rag")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name="logistics_rag",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    for filename, text in pdf_texts:
        split_text = text_splitter.split_text(text)
        for idx, chunk in enumerate(split_text):
            all_chunks.append({
                "id": f"{filename}_{idx}",
                "text": chunk,
                "source": filename
            })

    for chunk in all_chunks:
        embedding = embedding_model.encode(chunk["text"]).tolist()
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )

    return collection, len(all_chunks)


# =========================================================
# RETRIEVAL FUNCTION
# =========================================================

def retrieve_documents(collection, query: str, top_k: int = 3) -> list:
    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "document": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"]
        })
    return retrieved


def rag_lookup_factory(collection):
    """
    Returns a rag_lookup function bound to the given collection.
    This factory pattern is needed because Agno tools must be
    plain callables — we can't pass collection as a parameter.
    """
    def rag_lookup(query: str) -> str:
        docs = retrieve_documents(collection, query)
        return "\n\n".join([doc["document"] for doc in docs])
    return rag_lookup


# =========================================================
# AGENT BUILDER
# Built fresh per run (not cached) to pick up latest API key.
# =========================================================

def build_agent_team(rag_lookup_fn):
    """Build the three-agent team with the coordinator."""

    research_agent = Agent(
        name="Research Agent",
        model=OpenAIChat(id="gpt-4o", max_tokens=400),
        tools=[DuckDuckGoTools(), ArxivTools()],
        instructions="""
        You are a logistics SaaS market research analyst.
        Use DuckDuckGoTools for company news, hiring signals, and operational scale.
        Use ArxivTools for industry-wide market research and logistics digitization trends.
        Do NOT use ArxivTools to look up specific companies.
        Output concise structured findings.
        """,
        markdown=True
    )

    rag_agent = Agent(
        name="RAG Knowledge Agent",
        model=OpenAIChat(id="gpt-4o", max_tokens=350),
        tools=[rag_lookup_fn],
        instructions="""
        You are an internal logistics SaaS knowledge retrieval agent.
        Call rag_lookup with specific queries to retrieve:
        - pricing information
        - onboarding workflows
        - qualification rules
        - enterprise package information
        Use retrieved context carefully and cite the source.
        """,
        markdown=True
    )

    qualification_agent = Agent(
        name="Qualification Agent",
        model=OpenAIChat(id="gpt-4o", max_tokens=300),
        instructions="""
        You are a lead qualification specialist.
        Analyze: company size, operational complexity, urgency,
        potential ARR value, and enterprise qualification criteria.
        Produce: lead score (0-100), qualification reasoning,
        urgency classification, and recommended sales action.
        """,
        markdown=True
    )

    team = Team(
        name="Revenue Operations Coordinator",
        members=[research_agent, rag_agent, qualification_agent],
        model=OpenAIChat(id="gpt-4o", max_tokens=600),
        instructions="""
        Coordinate all agents. Combine external research,
        retrieved knowledge, and qualification reasoning.
        Generate: final lead intelligence report,
        enterprise recommendation, priority classification,
        and recommended next actions.
        Output a JSON summary at the end with keys:
        lead_score, urgency, recommended_action, estimated_arr.
        """,
        show_members_responses=True,
        markdown=True
    )

    return team


# =========================================================
# PDF TEXT EXTRACTOR
# =========================================================

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:18px;
                     font-weight:600; color:#2dd4bf;">LeadIntel</span>
        <span style="font-size:11px; color:#475569; margin-left:8px;">v1.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:11px;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.1em;">Configuration</p>',
                unsafe_allow_html=True)

    # API key input (only shown if not in secrets)
    try:
        st.secrets["OPENAI_API_KEY"]
        st.markdown(
            '<div style="background:#1e3a2f;border:1px solid #059669;'
            'border-radius:8px;padding:10px 14px;font-size:12px;color:#34d399;">'
            '🔐 API key loaded from secrets</div>',
            unsafe_allow_html=True
        )
    except (KeyError, FileNotFoundError):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is used only in this session and never stored."
        )
        st.session_state["api_key_input"] = api_key_input

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # PDF upload
    st.markdown('<p style="font-size:11px;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">'
                'Knowledge Base</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload logistics PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your logistics SaaS product docs, pricing sheets, and playbooks."
    )

    if uploaded_files:
        st.markdown(
            f'<div style="font-size:12px;color:#34d399;margin-top:6px;">'
            f'✓ {len(uploaded_files)} file(s) ready</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Pipeline info
    st.markdown('<p style="font-size:11px;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.1em;">Agent Pipeline</p>',
                unsafe_allow_html=True)

    for badge_class, label in [
        ("badge-research", "Research Agent"),
        ("badge-rag",      "RAG Knowledge Agent"),
        ("badge-qualify",  "Qualification Agent"),
        ("badge-coord",    "Coordinator"),
    ]:
        st.markdown(
            f'<span class="agent-badge {badge_class}">{label}</span>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:11px;color:#334155;line-height:1.6;">'
        'Built for Construction & Logistics SaaS — '
        'Agentic RAG Pipeline · Academic Portfolio Project</p>',
        unsafe_allow_html=True
    )


# =========================================================
# MAIN CONTENT
# =========================================================

st.markdown("""
<div class="header-banner">
    <p class="header-title">🚛 LeadIntel — Agentic RAG Pipeline</p>
    <p class="header-sub">
        Construction & Logistics SaaS · Sales Lead Qualification ·
        Multi-Agent System (Agno + OpenAI + ChromaDB)
    </p>
</div>
""", unsafe_allow_html=True)

# ── How it works strip ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
steps = [
    ("01", "Upload PDFs", "Product docs & pricing sheets become your knowledge base"),
    ("02", "Enter Lead",  "Paste inbound lead description or email content"),
    ("03", "Run Pipeline","3 agents retrieve, reason, and qualify in parallel"),
    ("04", "Get Brief",   "Receive a structured deal intelligence report"),
]
for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Step {num}</div>
            <div style="font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:4px;">{title}</div>
            <div style="font-size:12px;color:#475569;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Lead input ────────────────────────────────────────────────
st.markdown(
    '<p style="font-size:12px;font-weight:600;color:#475569;'
    'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">'
    'Inbound Lead Description</p>',
    unsafe_allow_html=True
)

default_lead = """We are a logistics company operating across 4 states.

We currently manage 85 trucks and 5 warehouse locations.

We are searching for:
- fleet optimization software
- warehouse visibility tools
- automated dispatch tracking

We need deployment within 60 days."""

lead_input = st.text_area(
    "Lead",
    value=default_lead,
    height=200,
    label_visibility="collapsed"
)

run_col, _ = st.columns([1, 3])
with run_col:
    run_button = st.button("▶  Run Pipeline", use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# =========================================================
# PIPELINE EXECUTION
# =========================================================

if run_button:

    # ── Validate API key ──────────────────────────────────────
    api_key = get_api_key()
    if not api_key:
        st.error(
            "⚠️ No API key found. Add your OpenAI key in the sidebar "
            "or configure it in Streamlit secrets."
        )
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    # ── Validate PDFs ─────────────────────────────────────────
    if not uploaded_files:
        st.warning(
            "⚠️ No PDFs uploaded. Upload your logistics knowledge base "
            "documents in the sidebar before running."
        )
        st.stop()

    if not lead_input.strip():
        st.warning("⚠️ Lead description is empty.")
        st.stop()

    # ── Build vector DB ───────────────────────────────────────
    with st.spinner("Building knowledge base from PDFs..."):
        pdf_texts = []
        for uf in uploaded_files:
            text = extract_pdf_text(uf)
            pdf_texts.append((uf.name, text))

        collection, num_chunks = build_vector_db(tuple(pdf_texts))

    st.markdown(
        f'<div style="font-size:12px;color:#34d399;margin-bottom:16px;">'
        f'✓ Knowledge base ready — {num_chunks} chunks from '
        f'{len(uploaded_files)} document(s)</div>',
        unsafe_allow_html=True
    )

    # ── RAG retrieval preview ─────────────────────────────────
    rag_lookup_fn = rag_lookup_factory(collection)
    retrieved_context = rag_lookup_fn(lead_input)

    with st.expander("📄 Retrieved Knowledge Base Context", expanded=False):
        st.markdown(
            f'<div class="output-box">{retrieved_context[:2000]}</div>',
            unsafe_allow_html=True
        )

    # ── Build agents ──────────────────────────────────────────
    with st.spinner("Initialising agent team..."):
        team = build_agent_team(rag_lookup_fn)

    # ── Construct prompt ──────────────────────────────────────
    prompt = f"""
Analyze the following inbound lead.

LEAD:
{lead_input}

INTERNAL KNOWLEDGE (retrieved from logistics knowledge base):
{retrieved_context}

Agent Instructions:
- Research Agent: Search for company profile, operational scale, hiring signals.
- RAG Knowledge Agent: Use the INTERNAL KNOWLEDGE above to match SaaS offerings,
  pricing tiers, and onboarding workflows relevant to this lead.
- Qualification Agent: Score this lead 0–100, classify urgency,
  estimate ARR, recommend sales action.

Produce a final structured report with a JSON summary at the end.
"""

    # ── Run pipeline with live output ─────────────────────────
    st.markdown(
        '<p style="font-size:12px;font-weight:600;color:#475569;'
        'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">'
        'Pipeline Output</p>',
        unsafe_allow_html=True
    )

    # Agent status indicators
    status_cols = st.columns(4)
    status_placeholders = []
    agent_labels = [
        ("badge-research", "Research Agent"),
        ("badge-rag",      "RAG Agent"),
        ("badge-qualify",  "Qualification"),
        ("badge-coord",    "Coordinator"),
    ]
    for col, (badge, label) in zip(status_cols, agent_labels):
        with col:
            ph = st.empty()
            ph.markdown(
                f'<span class="agent-badge {badge}" '
                f'style="opacity:0.35;">{label}</span>',
                unsafe_allow_html=True
            )
            status_placeholders.append((ph, badge, label))

    # Activate each badge in sequence as agents run
    def activate_badge(idx):
        ph, badge, label = status_placeholders[idx]
        ph.markdown(
            f'<span class="agent-badge {badge}">⚡ {label}</span>',
            unsafe_allow_html=True
        )

    def complete_badge(idx):
        ph, badge, label = status_placeholders[idx]
        ph.markdown(
            f'<span class="agent-badge {badge}">✓ {label}</span>',
            unsafe_allow_html=True
        )

    # Output display
    output_placeholder = st.empty()
    full_output = ""

    try:
        activate_badge(0)
        time.sleep(0.5)
        activate_badge(1)
        time.sleep(0.3)
        activate_badge(2)

        # Collect streamed response
        # agno's print_response streams to stdout; we capture via run()
        response = team.run(prompt)

        complete_badge(0)
        complete_badge(1)
        complete_badge(2)
        activate_badge(3)

        # Extract text from response
        if hasattr(response, "content"):
            full_output = response.content
        elif hasattr(response, "messages"):
            for msg in response.messages:
                if hasattr(msg, "content") and msg.content:
                    full_output += str(msg.content) + "\n"
        else:
            full_output = str(response)

        complete_badge(3)

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        st.info(
            "Common causes: invalid API key, rate limit hit, "
            "or DuckDuckGo timeout. Check your key and retry."
        )
        st.stop()

    # ── Display output ────────────────────────────────────────
    st.markdown(
        f'<div class="output-box">{full_output}</div>',
        unsafe_allow_html=True
    )

    # ── Try to extract and display JSON summary ───────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    import re
    json_match = re.search(r'\{[^{}]*"lead_score"[^{}]*\}', full_output, re.DOTALL)

    if json_match:
        try:
            summary = json.loads(json_match.group())

            st.markdown(
                '<p style="font-size:12px;font-weight:600;color:#475569;'
                'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">'
                'Summary Metrics</p>',
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)

            score = summary.get("lead_score", "N/A")
            score_class = (
                "score-hot" if isinstance(score, (int, float)) and score >= 70
                else "score-warm" if isinstance(score, (int, float)) and score >= 40
                else "score-cold"
            )

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Lead Score</div>
                    <div class="{score_class}">{score}</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Urgency</div>
                    <div class="metric-value">{summary.get("urgency", "—")}</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Recommended Action</div>
                    <div class="metric-value" style="font-size:14px;">
                        {summary.get("recommended_action", "—")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Est. ARR</div>
                    <div class="metric-value">{summary.get("estimated_arr", "—")}</div>
                </div>
                """, unsafe_allow_html=True)

        except json.JSONDecodeError:
            pass  # JSON parse failed — skip metrics display silently

    # ── Raw JSON expander ─────────────────────────────────────
    with st.expander("🔍 Full JSON Output", expanded=False):
        st.code(full_output, language="markdown")


# =========================================================
# EMPTY STATE (before first run)
# =========================================================

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #334155;">
        <div style="font-size:48px; margin-bottom:16px;">🚛</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:16px;
                    color:#475569; margin-bottom:8px;">
            Upload PDFs → Enter Lead → Run Pipeline
        </div>
        <div style="font-size:13px; color:#334155; max-width:400px; margin:0 auto;">
            Add your logistics knowledge base documents in the sidebar,
            then describe an inbound lead to generate a deal intelligence brief.
        </div>
    </div>
    """, unsafe_allow_html=True)
