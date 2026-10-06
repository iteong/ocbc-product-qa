"""Streamlit app: OCBC product Q&A for branch and contact-centre staff.

Answers come only from OCBC's public product data (pulled 2026-10-05, indicative), with citations, a refusal when
the data doesn't cover the question, and a source-and-freshness note on every answer.

ASSUMPTION (regulatory): this is an internal staff reference, not customer-facing advice.

Run:  streamlit run app.py
"""
import os
import re

import pandas as pd
import streamlit as st

from src.core import (DEFAULT_K, DEFAULT_MODEL, BM25Retriever, EmbeddingRetriever, HybridRetriever, answer,
                      build_chunks, get_client)

EVAL_SUMMARY = "data/eval/SUMMARY.md"
EVAL_RESULTS = {"BM25": "data/eval/results_bm25.csv", "Hybrid": "data/eval/results_hybrid.csv"}
EXAMPLES = [
    "What is the minimum balance for the 360 Account?",
    "Can I keep my savings in American dollars?",
    "What promotional rate does a S$30,000 time deposit for 12 months earn?",
    "Which accounts can a 16-year-old open on their own?",
    "What is the SIBOR-based home loan rate?",
    "What is the annual fee for the OCBC 365 Credit Card?",
]
STATUS_LABELS = {
    "answered": ("✅ Answered from product data", st.success),
    "partial": ("🟡 Partly answered: some of this isn't in the product data", st.warning),
    "not_in_data": ("⛔ Not in the product data", st.error),
    "ai_unavailable": ("⚠️ AI answer unavailable: showing the most relevant product sections instead", st.warning),
}

st.set_page_config(page_title="OCBC Product Q&A (staff)", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading product data…")
def load_chunks():
    """Chunks built once per app process."""
    return build_chunks()


@st.cache_resource
def load_bm25():
    """BM25 retriever (with synonyms) — the default."""
    return BM25Retriever(load_chunks())


@st.cache_resource(show_spinner="Loading embedding model (first use only)…")
def load_hybrid():
    """Hybrid retriever; loads the model2vec embeddings only when someone picks it."""
    chunks = load_chunks()
    return HybridRetriever(chunks, bm25=load_bm25(), embedding=EmbeddingRetriever(chunks))


@st.cache_resource
def load_client():
    """Anthropic client (key from .env, never displayed)."""
    return get_client()


def get_retriever(retriever_name):
    """The chosen retriever; if Hybrid can't load (e.g. embedding model unavailable), fall back to BM25."""
    if retriever_name == "Hybrid":
        try:
            return load_hybrid()
        except Exception as e:  # noqa: BLE001 - never crash the demo over the optional retriever
            st.warning(f"Hybrid retrieval unavailable ({type(e).__name__}); using BM25.")
    return load_bm25()


def get_answer(question, retriever_name, k):
    """Answer once per (question, retriever, k) per session, so re-renders don't re-bill the API.

    Only real answers are cached: an 'AI answer unavailable' fallback is retried next time.
    """
    cache = st.session_state.setdefault("answers", {})
    key = (question, retriever_name, k)
    if key in cache:
        return cache[key]
    try:
        client = load_client()
    except Exception:  # noqa: BLE001 - core.answer() turns a missing client into the fallback below
        client = None
    if client is None:
        load_client.clear()  # don't keep a failed client creation cached
    result = answer(question, get_retriever(retriever_name), k=k, client=client or _BrokenClient())
    if result["status"] != "ai_unavailable":
        cache[key] = result
    return result


class _BrokenClient:
    """Stand-in when the Anthropic client can't be created: any use raises, which answer() turns into the fallback."""

    def with_options(self, **_):
        raise RuntimeError("Anthropic client unavailable")


def render_citations_inline(text):
    """Show [chunk_id] citations as inline code so they stand out from the answer text."""
    return re.sub(r"\[([^\[\]]+#[^\[\]]+)\]", r" `\1`", text)


def render_freshness(note):
    """Freshness note: a warning box if any stale flag is present, otherwise an info box."""
    lines = note.split("\n")
    body = lines[0] + ("\n\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")
    (st.warning if "⚠" in note else st.info)(f"**Source & freshness**\n\n{body}")


def hits_table(hits):
    """Retrieved chunks as a table that shows WHY each was retrieved."""
    rows = []
    for h in hits:
        c = h["chunk"]
        rows.append({
            "rank": h["rank"], "chunk": c["chunk_id"], "score": h["score"],
            "bm25 rank": h.get("bm25_rank", h["rank"] if "matched_terms" in h else None),
            "embedding rank": h.get("embedding_rank"),
            "matched terms": ", ".join(h.get("matched_terms", [])),
            "synonyms added": ", ".join(h.get("expansions", [])),
            "stale flags": len(c["stale_flags"]),
        })
    df = pd.DataFrame(rows)
    return df.dropna(axis=1, how="all")


def render_result(result, k):
    """Render one answer result: status, answer, missing parts, citations, freshness, retrieved chunks."""
    label, box = STATUS_LABELS[result["status"]]
    box(label)
    if result["status"] == "ai_unavailable":
        render_fallback(result)
        return
    if result["answer"]:
        st.markdown(render_citations_inline(result["answer"]))
    if result["refusal_reason"]:
        st.markdown(f"**Not covered by the data:** {result['refusal_reason']}")
    if result["status"] == "not_in_data":
        st.caption("Check ocbc.com or the product team. Don't answer the customer from memory.")

    if result["citations"]:
        st.markdown("**Cited sources**")
        for c in result["citations"]:
            st.markdown(f"- {c['product_name']} · {c['section']} · `{c['chunk_id']}` · source file `{c['source_file']}`")
    if result["dropped_citations"]:
        st.caption(f"Removed citations not in the retrieved data: {result['dropped_citations']}")
    render_freshness(result["freshness_note"])

    with st.expander(f"Why these sources? Top {k} retrieved chunks"):
        st.dataframe(hits_table(result["hits"]), hide_index=True, width="stretch")
        for h in result["hits"]:
            st.markdown(f"**{h['rank']}. `{h['chunk']['chunk_id']}`**")
            st.code(h["chunk"]["text"], language=None, wrap_lines=True)
    if result["usage"]:
        st.caption(f"{result['model']} · {result['usage']['input_tokens']} input / "
                   f"{result['usage']['output_tokens']} output tokens")
    else:
        st.caption("Refused before calling the model: no product data matched the question.")


def render_fallback(result):
    """No AI answer (API error, timeout or unusable output): show the top retrieved sections with citations."""
    st.markdown("The AI service didn't respond in time or returned an error. These are the product sections that "
                "best match the question. Read them directly; nothing below was written by AI.")
    for sec in result["fallback_sections"]:
        st.markdown(f"**{sec['product_name']} · {sec['section']}**  \n`{sec['chunk_id']}` · source file "
                    f"`{sec['source_file']}`")
        st.code(sec["text"], language=None, wrap_lines=True)
    render_freshness(result["freshness_note"])
    st.caption(f"AI answer unavailable ({result['error']}). Try again in a moment.")


def ask_tab(retriever_name, k):
    """The question-and-answer tab."""
    st.markdown("Ask about OCBC current, savings, foreign-currency and time-deposit accounts, home loans and unit trusts.")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.question = ex
    with st.form("ask"):
        question = st.text_input("Question", key="question", placeholder="e.g. What is the fall-below fee on the USD Current Account?")
        submitted = st.form_submit_button("Ask", type="primary")
    question = (question or "").strip()
    if not question or not (submitted or question in EXAMPLES):
        return
    # Last line of defence: answer() already falls back on AI errors, but nothing should show a stack trace on screen.
    try:
        with st.spinner("Searching product data and drafting an answer…"):
            result = get_answer(question, retriever_name, k)
        render_result(result, k)
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong ({type(e).__name__}). Please try again, or check ocbc.com directly.")


def eval_tab():
    """The evaluation tab: latest BM25 vs hybrid summary and per-question results."""
    if not os.path.exists(EVAL_SUMMARY):
        st.info("No evaluation yet. Run `python -m src.eval`.")
        return
    st.caption("Generated by `python -m src.eval`. The question set was also used to tune synonyms and k, "
               "so these scores are optimistic until checked on held-out questions.")
    with open(EVAL_SUMMARY, encoding="utf-8") as f:
        st.markdown(f.read())
    st.subheader("Per-question results")
    which = st.radio("Retriever", list(EVAL_RESULTS), horizontal=True)
    if os.path.exists(EVAL_RESULTS[which]):
        df = pd.read_csv(EVAL_RESULTS[which])
        only_wrong = st.checkbox("Show only wrong answers")
        if only_wrong:
            df = df[~df.correct]
        st.dataframe(df[["id", "type", "status", "correct", "facts_found", "citations", "missing", "answer"]],
                     hide_index=True, width="stretch")


def sidebar():
    """Settings and the standing data notice. Returns (retriever name, k)."""
    with st.sidebar:
        st.header("Settings")
        retriever_name = st.radio("Retrieval", ["BM25", "Hybrid"], index=0,
                                  help="BM25 (keywords + synonyms) scored best in evaluation and handles rate "
                                       "tables well. Hybrid adds model2vec embeddings: better on loose wording, "
                                       "worse on figures.")
        k = st.slider("Chunks sent to the model", 3, 12, DEFAULT_K)
        st.caption(f"Model: `{os.getenv('OCBC_QA_MODEL', DEFAULT_MODEL)}`")
        st.divider()
        st.markdown("**About the data**")
        st.caption("OCBC public product API data pulled **5 Oct 2026**. Rates and fees are indicative and may be "
                   "outdated. Verify on ocbc.com before quoting to a customer. Credit cards, current FD rates and "
                   "promotions are not in the data.")
        st.caption("Internal staff reference only, not financial advice. No customer data is used.")
    return retriever_name, k


def main():
    """App entry point."""
    st.title("🏦 OCBC Product Q&A")
    st.caption("For branch and contact-centre staff · answers only from OCBC's public product data, with sources")
    retriever_name, k = sidebar()
    ask, evaluation = st.tabs(["Ask", "Evaluation"])
    with ask:
        ask_tab(retriever_name, k)
    with evaluation:
        eval_tab()


main()
