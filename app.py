"""
SciDiscover — Streamlit GUI

Run:
    streamlit run app.py

The CLI (python -m scidiscover.eval.run ...) is completely unaffected.
"""

import copy
import time

import streamlit as st
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SciDiscover",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 SciDiscover")
st.caption("Multi-agent scientific literature synthesis")

# ---------------------------------------------------------------------------
# Model options
# ---------------------------------------------------------------------------

MODEL_OPTIONS = {
    "DeepSeek-R1 32B (Local HPC)": "local-deepseek-r1-distill-qwen-32b",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
}

CLAIM_STATUS_COLORS = {
    "SUPPORTED":   "🟢",
    "UNSUPPORTED": "🔴",
    "CONFLICT":    "🟡",
    "UNKNOWN":     "⚪",
}

# ---------------------------------------------------------------------------
# Sidebar — pipeline config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline settings")

    model_label = st.selectbox("LLM model", list(MODEL_OPTIONS.keys()))
    model_id = MODEL_OPTIONS[model_label]

    max_tokens = st.slider("Max tokens", min_value=1024, max_value=8192,
                           value=4096, step=256)
    top_k = st.slider("Retrieve top-k chunks", min_value=5, max_value=40,
                      value=20, step=5)
    use_reranker = st.toggle("Reranker", value=False)
    use_verifier = st.toggle("Verifier", value=False)
    max_workers = st.slider("Summariser workers", min_value=1, max_value=16,
                            value=4, step=1,
                            help="1 = sequential. Use 3 for Gemini (rate limits), 8+ for local vLLM.")

    st.divider()
    st.caption("Config: `configs/demo.yaml` (read-only)")

# ---------------------------------------------------------------------------
# Load base config once
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_base_config():
    with open("configs/demo.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Agent factory — cached per (model, reranker) so swapping model rebuilds
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _build_agents(model: str, max_tok: int, reranker_on: bool, workers: int = 4):
    from scidiscover.agents.retriever import Retriever
    from scidiscover.agents.retriever_agent import RetrieverAgent
    from scidiscover.agents.summarizer import SummarizerAgent
    from scidiscover.agents.synthesizer import SynthesizerAgent
    from scidiscover.agents.reranker import RerankerAgent
    from scidiscover.agents.verifier import VerifierAgent
    from utils.llm_client import configure_llm

    cfg = copy.deepcopy(_load_base_config())
    cfg["llm"]["model"] = model
    cfg["llm"]["max_tokens"] = max_tok
    cfg["reranker"]["enabled"] = reranker_on

    configure_llm(cfg["llm"]["model"], cfg["llm"]["max_tokens"])

    retriever = Retriever(
        chunks_input=cfg["retrieval"]["chunks_input"],
        index_output=cfg["retrieval"]["index_output"],
        chunk_ids_output=cfg["retrieval"]["chunk_ids_output"],
        traces_output=cfg["retrieval"]["traces_output"],
        model_name=cfg["retrieval"]["model_name"],
        papers_input=cfg["retrieval"]["papers_input"],
    )

    agents = {
        "retriever_agent": RetrieverAgent(retriever),
        "summarizer_agent": SummarizerAgent(
            prompt_path=cfg["summarizer"]["prompt_path"],
            traces_output=cfg["summarizer"]["traces_output"],
            output_path=cfg["summarizer"]["output_path"],
            max_workers=workers,
        ),
        "synthesizer_agent": SynthesizerAgent(
            prompt_path=cfg["synthesizer"]["prompt_path"],
            traces_output=cfg["synthesizer"]["traces_output"],
            output_path=cfg["synthesizer"]["output_path"],
        ),
        "verifier_agent": VerifierAgent(
            prompt_path=cfg["verifier"]["prompt_path"],
            traces_output=cfg["verifier"]["traces_output"],
            verification_output=cfg["verifier"]["verification_output"],
            answers_output=cfg["verifier"]["answers_output"],
            min_citation_coverage=cfg["verifier"].get("min_citation_coverage", 0.60),
            min_support_rate=cfg["verifier"].get("min_support_rate", 0.40),
        ),
    }

    if reranker_on:
        rcfg = cfg["reranker"]
        agents["reranker_agent"] = RerankerAgent(
            model_name=rcfg["model_name"],
            top_k=rcfg.get("top_k", 10),
            min_score=rcfg.get("min_score", 0.0),
        )

    return agents


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _render_retrieved(evidence_pack: dict):
    chunks = evidence_pack["chunks"]
    st.markdown(f"**{len(chunks)} chunks** retrieved across "
                f"**{len({c['paper_id'] for c in chunks})} papers**")
    rows = [
        {
            "Paper": c["paper_id"].split("_")[-1] if "_" in c["paper_id"] else c["paper_id"][:30],
            "Section": c["section"],
            "Score": round(c["score"], 3),
            "Preview": c["text"][:120] + "…",
        }
        for c in chunks
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_summaries(summaries: list):
    good = [s for s in summaries if not s["summary_text"].startswith("[LLM ERROR")
            and not s["summary_text"].lower().startswith("insufficient evidence")]
    st.markdown(f"**{len(good)}/{len(summaries)}** papers summarised successfully")
    for s in summaries:
        label = s["paper_id"].split("_")[-1] if "_" in s["paper_id"] else s["paper_id"][:40]
        url = s.get("url", "")
        header = f"[{label}]({url})" if url else label
        with st.expander(header, expanded=False):
            st.markdown(s["summary_text"])


def _render_synthesis(synthesis: dict):
    draft = synthesis.get("draft_answer", "")
    claims = synthesis.get("key_claims", [])
    st.markdown(draft)
    if claims:
        st.markdown(f"**{len(claims)} key claims extracted**")
        for i, c in enumerate(claims, 1):
            cids = ", ".join(c.get("citation_ids", [])) or "none"
            st.markdown(f"{i}. {c['claim']}  \n   *Citations:* `{cids}`")


def _render_verification(verified: dict):
    vs = verified.get("verification_summary", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Citation coverage", f"{vs.get('citation_coverage', 0):.0%}")
    col2.metric("Support rate", f"{vs.get('support_rate', 0):.0%}")
    col3.metric("Total claims", vs.get("total_claims", 0))
    col4.metric("Supported", vs.get("supported", 0))

    for c in verified.get("key_claims", []):
        status = c.get("status", "UNKNOWN")
        icon = CLAIM_STATUS_COLORS.get(status, "⚪")
        with st.expander(f"{icon} {c['claim'][:100]}", expanded=False):
            st.markdown(f"**Status:** {status}")
            if c.get("notes"):
                st.markdown(f"**Notes:** {c['notes']}")
            cids = c.get("citation_ids", [])
            if cids:
                st.markdown("**Citations:** " + ", ".join(f"`{cid}`" for cid in cids))


def _render_final_answer(answer: str, abstained: bool):
    if abstained:
        st.warning(answer)
    else:
        st.success("**Answer**")
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Main — query input and pipeline run
# ---------------------------------------------------------------------------

query = st.text_area("Research query", height=100,
                     placeholder="e.g. What is the role of GRK4 in essential hypertension?")

run_clicked = st.button("Run pipeline", type="primary", disabled=not query.strip())

_ABSTAIN_MSG = "Insufficient evidence in retrieved corpus to answer reliably."

if run_clicked and query.strip():
    t0 = time.time()

    try:
        agents = _build_agents(model_id, max_tokens, use_reranker, max_workers)
    except Exception as e:
        st.error(f"Failed to initialise agents: {e}")
        st.stop()

    # Stage 1 — Retrieval
    with st.status("Retrieving papers…", expanded=True) as status:
        evidence_pack = agents["retriever_agent"].run(query, k=top_k)
        n_chunks = len(evidence_pack["chunks"])
        n_papers = len({c["paper_id"] for c in evidence_pack["chunks"]})
        st.markdown(f"Found **{n_chunks} chunks** across **{n_papers} papers**")
        status.update(label=f"Retrieved — {n_chunks} chunks / {n_papers} papers",
                      state="complete")
    _render_retrieved(evidence_pack)

    # Stage 2 — Reranker (optional)
    if use_reranker and "reranker_agent" in agents:
        with st.status("Reranking…", expanded=True) as status:
            evidence_pack = agents["reranker_agent"].run(evidence_pack)
            n_reranked = len(evidence_pack["chunks"])
            st.markdown(f"Reranked to **{n_reranked} chunks**")
            status.update(label=f"Reranked — {n_reranked} chunks", state="complete")

    # Stage 3 — Summarisation
    with st.status("Summarising papers…", expanded=True) as status:
        summaries = agents["summarizer_agent"].run(evidence_pack)
        good = sum(1 for s in summaries
                   if not s["summary_text"].startswith("[LLM ERROR")
                   and not s["summary_text"].lower().startswith("insufficient evidence"))
        st.markdown(f"Summarised **{good}/{len(summaries)}** papers successfully")
        status.update(label=f"Summarised — {good}/{len(summaries)} papers",
                      state="complete")
    _render_summaries(summaries)

    # Stage 4 — Synthesis
    with st.status("Synthesising answer…", expanded=True) as status:
        synthesis = agents["synthesizer_agent"].run({
            "query": query,
            "paper_summaries": summaries,
        })
        n_claims = len(synthesis.get("key_claims", []))
        st.markdown(f"Generated draft answer with **{n_claims} key claims**")
        status.update(label=f"Synthesised — {n_claims} claims", state="complete")
    _render_synthesis(synthesis)

    # Stage 5 — Verification (optional)
    if use_verifier:
        with st.status("Verifying claims…", expanded=True) as status:
            verified = agents["verifier_agent"].run({
                "synthesis": synthesis,
                "evidence_pack": evidence_pack,
            })
            vs = verified.get("verification_summary", {})
            st.markdown(
                f"Support rate **{vs.get('support_rate', 0):.0%}** · "
                f"Citation coverage **{vs.get('citation_coverage', 0):.0%}**"
            )
            status.update(label=(
                f"Verified — support {vs.get('support_rate', 0):.0%} · "
                f"coverage {vs.get('citation_coverage', 0):.0%}"
            ), state="complete")
        _render_verification(verified)

        draft = verified["final_answer"]
        abstained = draft == _ABSTAIN_MSG
    else:
        draft = synthesis.get("draft_answer", "")
        abstained = (
            not draft
            or draft.startswith("ABSTAIN")
            or draft.startswith("[")
            or "paper summaries" in draft.lower()
        )
        if abstained:
            draft = _ABSTAIN_MSG

    st.divider()
    _render_final_answer(draft, abstained)

    latency = round(time.time() - t0, 1)
    st.caption(f"Completed in {latency}s · model: {model_id} · "
               f"reranker: {'on' if use_reranker else 'off'} · "
               f"verifier: {'on' if use_verifier else 'off'}")
