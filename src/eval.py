"""Evaluation for the product Q&A assistant.

Step 2: load and validate the hand-written question set (data/eval/questions.json).
Step 7: compare BM25-only vs hybrid retrieval.
  - Retrieval metrics (no LLM, deterministic): section hit@1/3/5/8, MRR, product hit@8, section recall@8.
  - Answer metrics (calls Claude): status/refusal accuracy, required facts present, expected section cited,
    stale warning present, invalid citations dropped, tokens and estimated cost.

Run from the project folder:
  python -m src.eval                    # validate + retrieval + answers (about 88 Claude calls)
  python -m src.eval --retrieval-only   # no API calls
Writes data/eval/retrieval_results.csv, data/eval/results_<retriever>.csv and data/eval/SUMMARY.md.
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.data_gen import build_documents

QUESTIONS_PATH = "data/eval/questions.json"
QUESTION_TYPES = {"factual", "paraphrase", "numeric", "cross_product", "unanswerable", "stale"}
REQUIRED_KEYS = {"id", "type", "question", "expected_products", "expected_sections",
                 "must_contain", "should_refuse", "expect_stale_flag", "notes"}


def load_questions(path=QUESTIONS_PATH):
    """Load the evaluation questions (one dict per question)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def section_index(docs):
    """Map 'doc_id#Section title' -> (product_name, section text) for every section in the docs."""
    return {f"{d['doc_id']}#{title}": (d["product_name"], text)
            for d in docs for title, text in d["sections"].items()}


def check_question(q, doc_ids, sections):
    """Return a list of problems with one question (empty list = valid)."""
    problems = []
    missing = REQUIRED_KEYS - q.keys()
    if missing:
        return [f"missing keys {sorted(missing)}"]
    if q["type"] not in QUESTION_TYPES:
        problems.append(f"unknown type {q['type']!r}")
    problems += [f"unknown product {p!r}" for p in q["expected_products"] if p not in doc_ids]
    problems += [f"unknown section {s!r}" for s in q["expected_sections"] if s not in sections]
    problems += [f"section {s!r} not under an expected product" for s in q["expected_sections"]
                 if s.split("#")[0] not in q["expected_products"]]
    if q["should_refuse"]:
        if q["must_contain"]:
            problems.append("refusal question should not have must_contain")
    elif not (q["expected_sections"] and q["must_contain"]):
        problems.append("answerable question needs expected_sections and must_contain")
    # Grounding check: every expected fact must literally appear in an expected section (or its product name),
    # so the eval can't reward an answer for facts that aren't in the source data.
    grounding = " ".join(f"{name} {text}" for s in q["expected_sections"] if s in sections
                         for name, text in [sections[s]]).lower()
    problems += [f"must_contain {m!r} not found in expected sections" for m in q["must_contain"]
                 if m.lower() not in grounding]
    return problems


def validate_questions(questions, docs=None):
    """Validate every question against the documents. Returns {question_id: [problems]} for failures only."""
    docs = build_documents() if docs is None else docs
    doc_ids, sections = {d["doc_id"] for d in docs}, section_index(docs)
    failures = {q.get("id", "?"): p for q in questions if (p := check_question(q, doc_ids, sections))}
    ids = [q.get("id") for q in questions]
    for dup in {i for i in ids if ids.count(i) > 1}:
        failures.setdefault(dup, []).append("duplicate id")
    return failures


# ---------------------------------------------------------------------------------------------------------------
# Step 7: retrieval metrics (no LLM)
# ---------------------------------------------------------------------------------------------------------------

EVAL_DIR = "data/eval"
KS = (1, 3, 5, 8)
# USD per 1M tokens (input, output), from Anthropic's published API pricing as cached on 2026-09-25.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-4-5": (1.0, 5.0)}


def section_key(chunk):
    """'doc_id#Section' for a chunk, so every part of a split section matches the question's expected section."""
    return f"{chunk['doc_id']}#{chunk['section']}"


def retrieval_row(q, retriever, max_k=max(KS)):
    """Retrieval metrics for one answerable question and one retriever."""
    hits = retriever.search(q["question"], k=max_k)
    sections = [section_key(h["chunk"]) for h in hits]
    products = [h["chunk"]["doc_id"] for h in hits]
    expected = set(q["expected_sections"])
    first = next((i + 1 for i, s in enumerate(sections) if s in expected), None)
    row = {"id": q["id"], "type": q["type"], "retriever": retriever.name, "first_rank": first,
           "rr": 1 / first if first else 0.0,
           "product_hit@8": bool(set(products) & set(q["expected_products"])),
           "section_recall@8": len(set(sections) & expected) / len(expected),
           "top3": " | ".join(sections[:3])}
    row.update({f"hit@{k}": bool(first and first <= k) for k in KS})
    return row


def evaluate_retrieval(retrievers, questions):
    """Run retrieval metrics for every answerable question and retriever. Returns a DataFrame (one row each)."""
    answerable = [q for q in questions if not q["should_refuse"]]
    return pd.DataFrame([retrieval_row(q, r) for r in retrievers for q in answerable])


def summarise_retrieval(df):
    """Overall and per-type retrieval scores per retriever (means; hit rates as fractions)."""
    cols = [f"hit@{k}" for k in KS] + ["rr", "product_hit@8", "section_recall@8"]
    overall = df.groupby("retriever", sort=False)[cols].mean().rename(columns={"rr": "MRR"})
    by_type = df.pivot_table(index="type", columns="retriever", values="hit@3", aggfunc="mean", sort=False)
    return overall, by_type


def keyword_confidence_table(questions, bm25):
    """Top BM25 score per question, to check the pre-LLM refusal threshold against real questions."""
    from src.core import keyword_confidence
    return pd.DataFrame([{"id": q["id"], "should_refuse": q["should_refuse"],
                          "top_bm25": keyword_confidence(q["question"], bm25)} for q in questions])


# ---------------------------------------------------------------------------------------------------------------
# Step 7: answer metrics (calls Claude)
# ---------------------------------------------------------------------------------------------------------------

def normalise_text(text):
    """Lowercase, drop thousands separators and turn hyphens into spaces, so '3,000' matches 'S$3000'."""
    text = text.lower().replace("-", " ")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def grade_answer(q, result):
    """Grade one answer against its question. Returns a flat dict of booleans/counts for the results CSV."""
    refused = result["status"] == "not_in_data"
    text = normalise_text(result["answer"])
    facts = [normalise_text(m) in text for m in q["must_contain"]]
    cited_sections = {f"{c['chunk_id'].split('#')[0]}#{c['section']}" for c in result["citations"]}
    usage = result["usage"] or {"input_tokens": 0, "output_tokens": 0}
    row = {
        "id": q["id"], "type": q["type"], "should_refuse": q["should_refuse"], "status": result["status"],
        "refusal_correct": refused == q["should_refuse"],
        "facts_found": f"{sum(facts)}/{len(facts)}" if facts else "",
        "facts_ok": (not refused and all(facts)) if not q["should_refuse"] else None,
        "cited_expected_section": bool(cited_sections & set(q["expected_sections"])) if not q["should_refuse"] else None,
        "stale_warning_ok": ("⚠" in result["freshness_note"]) if q["expect_stale_flag"] else None,
        "dropped_citations": len(result["dropped_citations"]),
        "llm_called": result["usage"] is not None,
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "missing": result["refusal_reason"], "answer": result["answer"],
        "citations": "; ".join(c["chunk_id"] for c in result["citations"]),
        "retrieved": "; ".join(h["chunk"]["chunk_id"] for h in result["hits"]),
    }
    # One overall verdict: right refusal decision, and for answerable questions every required fact present.
    row["correct"] = row["refusal_correct"] and (q["should_refuse"] or bool(row["facts_ok"]))
    return row


def evaluate_answers(retriever, questions, client, model=None, workers=6):
    """Answer every question with one retriever (in parallel threads) and grade each. Returns a DataFrame."""
    from src.core import answer

    def run(q):
        return grade_answer(q, answer(q["question"], retriever, client=client, model=model))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, questions))
    df = pd.DataFrame(rows)
    df.insert(2, "retriever", retriever.name)
    return df


def summarise_answers(df, model):
    """Headline answer metrics per retriever."""
    price_in, price_out = PRICES.get(model, (float("nan"), float("nan")))
    out = {}
    for name, g in df.groupby("retriever", sort=False):
        answerable, refuse = g[~g.should_refuse], g[g.should_refuse]
        said_no = g.status == "not_in_data"
        out[name] = {
            "overall correct": g.correct.mean(),
            "facts ok (answerable)": answerable.facts_ok.mean(),
            "cited expected section": answerable.cited_expected_section.mean(),
            "refusal recall": refuse.refusal_correct.mean(),  # should refuse -> did
            "refusal precision": (said_no & g.should_refuse).sum() / max(said_no.sum(), 1),
            "false refusals": int((said_no & ~g.should_refuse).sum()),
            "stale warning ok": g.stale_warning_ok.dropna().astype(bool).mean(),
            "partial answers": int((g.status == "partial").sum()),
            "AI unavailable (errors)": int((g.status == "ai_unavailable").sum()),
            "dropped citations": int(g.dropped_citations.sum()),
            "LLM calls": int(g.llm_called.sum()),
            "est. cost USD": (g.input_tokens.sum() * price_in + g.output_tokens.sum() * price_out) / 1e6,
        }
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------------------

def md_table(df, fmt="{:.2f}"):
    """Render a DataFrame as a Markdown table (no tabulate dependency)."""
    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else fmt.format(v)
        return str(v)
    header = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([header, sep] + rows)


def failure_lines(answers):
    """One line per wrong answer, to make failures easy to inspect."""
    lines = []
    for _, r in answers[~answers.correct].iterrows():
        why = "wrong refusal decision" if not r.refusal_correct else f"facts {r.facts_found}"
        lines.append(f"- **{r.retriever} / {r.id}** ({r.type}, status={r.status}): {why}. "
                     f"Retrieved: {r.retrieved.split('; ')[:3]}")
    return lines


def write_summary(path, retrieval_overall, retrieval_by_type, grid, thresholds, answers=None, answer_summary=None,
                  model=None):
    """Write the evaluation summary as Markdown."""
    parts = ["# Evaluation: BM25 vs hybrid", "",
             "Product data pulled 2026-10-05 (indicative). Questions: data/eval/questions.json. "
             "Small hand-written set (44 questions): treat differences of 1-2 questions as noise.", "",
             "## Retrieval (35 answerable questions, no LLM)", "", md_table(retrieval_overall), "",
             "### Section hit@3 by question type", "", md_table(retrieval_by_type), "",
             "### Hybrid weight sensitivity (hit@3 / MRR)", "",
             "Tuned on the same questions it is scored on, so this shows sensitivity, not a validated setting.", "",
             md_table(grid), "",
             "### Pre-LLM refusal threshold (top BM25 score)", "", md_table(thresholds), ""]
    if answers is not None:
        parts += [f"## Answers ({model}, all 44 questions)", "", md_table(answer_summary), "",
                  "### Overall correct by question type", "",
                  md_table(answers.pivot_table(index="type", columns="retriever", values="correct",
                                               aggfunc="mean", sort=False)), "",
                  "### Wrong answers", ""] + failure_lines(answers) + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def run_evaluation(retrieval_only=False, model=None):
    """Validate questions, then evaluate BM25 and hybrid on retrieval and (optionally) answers. Prints and saves results."""
    from src.core import (DEFAULT_MODEL, BM25Retriever, EmbeddingRetriever, HybridRetriever, build_chunks,
                          get_client)
    import os

    questions = load_questions()
    failures = validate_questions(questions)
    if failures:
        sys.exit(f"Question set has problems: {failures}")
    model = model or os.getenv("OCBC_QA_MODEL", DEFAULT_MODEL)

    chunks = build_chunks()
    bm25, emb = BM25Retriever(chunks), EmbeddingRetriever(chunks)
    hybrid = HybridRetriever(chunks, bm25, emb)

    # Retrieval (embedding-only included as a reference point for why hybrid behaves as it does).
    bm25_plain = BM25Retriever(chunks, expand=False)  # reference: the step 7 baseline tokenizer, no synonyms
    rdf = evaluate_retrieval([bm25, hybrid, emb, bm25_plain], questions)
    rdf.to_csv(f"{EVAL_DIR}/retrieval_results.csv", index=False)
    overall, by_type = summarise_retrieval(rdf)
    print("Retrieval, 35 answerable questions (fractions):\n" + overall.round(2).to_string())
    print("\nSection hit@3 by type:\n" + by_type.round(2).to_string())

    grid_rows = {}
    for wb, we in [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)]:
        h = HybridRetriever(chunks, bm25, emb, bm25_weight=wb, embedding_weight=we)
        g = evaluate_retrieval([h], questions)
        grid_rows[f"bm25 x{wb}, emb x{we}"] = {"hit@3": g["hit@3"].mean(), "MRR": g["rr"].mean(),
                                                "paraphrase hit@3": g[g.type == "paraphrase"]["hit@3"].mean(),
                                                "numeric hit@3": g[g.type == "numeric"]["hit@3"].mean()}
    grid = pd.DataFrame(grid_rows).T
    print("\nHybrid weight sensitivity:\n" + grid.round(2).to_string())

    conf = keyword_confidence_table(questions, bm25)
    thresholds = conf.groupby("should_refuse")["top_bm25"].describe()[["count", "min", "50%", "max"]]
    thresholds.index = thresholds.index.map({False: "answerable", True: "should refuse"})
    print("\nTop BM25 score by question kind:\n" + thresholds.round(2).to_string())

    answers = answer_summary = None
    if not retrieval_only:
        client = get_client()
        frames = []
        for r in (bm25, hybrid):
            print(f"\nAnswering {len(questions)} questions with {r.name} + {model} ...", flush=True)
            df = evaluate_answers(r, questions, client, model=model)
            df.to_csv(f"{EVAL_DIR}/results_{r.name}.csv", index=False)
            frames.append(df)
        answers = pd.concat(frames, ignore_index=True)
        answer_summary = summarise_answers(answers, model)
        print("\nAnswers:\n" + answer_summary.round(3).to_string())
        print("\nOverall correct by type:\n" + answers.pivot_table(index="type", columns="retriever", values="correct",
                                                                    aggfunc="mean", sort=False).round(2).to_string())
        print("\nWrong answers:\n" + "\n".join(failure_lines(answers)))

    write_summary(f"{EVAL_DIR}/SUMMARY.md", overall, by_type, grid, thresholds, answers, answer_summary, model)
    print(f"\nWrote {EVAL_DIR}/SUMMARY.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrieval-only", action="store_true", help="skip Claude calls")
    parser.add_argument("--validate-only", action="store_true", help="only validate the question set")
    parser.add_argument("--model", help="Claude model id (default: OCBC_QA_MODEL or claude-sonnet-5-5)")
    args = parser.parse_args()
    if args.validate_only:
        qs = load_questions()
        problems = validate_questions(qs)
        print(f"{len(qs)} questions; {len(problems)} with problems")
        for qid, probs in problems.items():
            print(f"  {qid}: " + "; ".join(probs))
        print("By type:", dict(Counter(q["type"] for q in qs)))
    else:
        run_evaluation(retrieval_only=args.retrieval_only, model=args.model)
