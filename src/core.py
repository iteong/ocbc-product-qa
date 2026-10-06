"""Core logic for the product Q&A assistant.

Step 3: section-level chunking with context headers and staleness flags.
Step 4: BM25 keyword retrieval that reports which query terms matched (the WHY).
Step 5: Claude answers only from retrieved chunks, with citations, refusals and a code-built freshness note.
Step 6 (brought forward): model2vec embedding retrieval and a BM25 + embedding hybrid (Reciprocal Rank Fusion).

Run from the project folder:  python -m src.core ["your question"]
"""
import hashlib
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.data_gen import build_documents

MAX_CHUNK_CHARS = 900  # body length above which a section is split; the corpus is small, so most sections fit whole

# Staleness rules: (regex on chunk body, flag shown to staff). These are assumptions about the data, not facts
# from OCBC, and are surfaced in the freshness note on every answer that cites the chunk.
STALE_RULES = [
    # ASSUMPTION: SIBOR was discontinued in Singapore (replaced by SORA), so SIBOR-pegged packages are likely
    # no longer offered as written.
    (r"\bSIBOR\b", "References SIBOR, which has been discontinued in Singapore; these packages may no longer be offered."),
    (r"\bas of (\w+ )?(19|20)\d\d\b", "Contains figures stated 'as of' a past date."),
    (r"back[ -]?tested", "Performance is back-tested, not actual returns, and is not a forecast."),
    (r"Nov16", "Contains a link tagged Nov16 (a 2016 campaign); the page may have changed."),
    # ASSUMPTION: promotional rates are time-limited, so they are the most likely figures to be out of date.
    (r"\bPromotional\b", "Promotional rates are time-limited and may have ended."),
]


def stale_flags(text):
    """Return the staleness flags whose rule matches the text (order follows STALE_RULES)."""
    return [flag for pattern, flag in STALE_RULES if re.search(pattern, text, re.IGNORECASE)]


def _indent(line):
    """Number of leading spaces on a line."""
    return len(line) - len(line.lstrip(" "))


def _is_bullet(line):
    """True if the line is a '- ' bullet (at any indent)."""
    return line.lstrip(" ").startswith("- ")


def _blocks(lines):
    """Group lines into top-level blocks that must stay together.

    A top-level non-bullet line is a heading that owns the top-level bullets after it (e.g. a home loan
    package name and its bullet points). Otherwise each top-level bullet owns its indented children
    (e.g. a TD rate tier and its tenors).
    """
    blocks = []
    for line in lines:
        top = _indent(line) == 0
        starts_new = top and (not _is_bullet(line) or not blocks or _is_bullet(blocks[-1][0]))
        if starts_new or not blocks:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def _dedent(lines):
    """Remove the common leading indentation from a list of lines."""
    n = min((_indent(l) for l in lines if l.strip()), default=0)
    return [l[n:] for l in lines]


def _label(line):
    """Turn a heading or bullet line into a short breadcrumb label: '- Singapore Dollar:' -> 'Singapore Dollar'."""
    return line.strip().removeprefix("- ").rstrip(":").strip()


def split_section(text, max_chars=MAX_CHUNK_CHARS, path=()):
    """Split section text into parts of at most ~max_chars without breaking a block.

    Returns a list of (breadcrumb, body) pairs. If a single block is still too big, it is split into its
    children and its heading becomes part of the breadcrumb, so each part keeps its context
    (e.g. 'Promotional Interest Rates (% per year) > Singapore Dollar').
    """
    if len(text) <= max_chars:
        return [(" > ".join(path), text)]
    parts, current = [], []

    def flush():
        if current:
            parts.append((" > ".join(path), "\n".join(current)))
            current.clear()

    for block in _blocks(text.split("\n")):
        block_text = "\n".join(block)
        if len(block_text) > max_chars and len(block) > 1:
            flush()
            parts += split_section("\n".join(_dedent(block[1:])), max_chars, path + (_label(block[0]),))
            continue
        if current and len("\n".join(current)) + 1 + len(block_text) > max_chars:
            flush()
        current += block
    flush()
    return parts


def chunk_document(doc, max_chars=MAX_CHUNK_CHARS):
    """Turn one document into chunks: one per section, or several '-1', '-2', ... parts for long sections.

    Every chunk's searchable text starts with a context header naming the product and section, because
    section text alone often doesn't mention the product (e.g. a bare 'S$3,000' minimum balance).
    """
    chunks = []
    for section, text in doc["sections"].items():
        parts = split_section(text, max_chars)
        for i, (breadcrumb, body) in enumerate(parts, start=1):
            suffix = f"-{i}" if len(parts) > 1 else ""
            header = f"{doc['product_name']} ({doc['category']}) — {section}"
            if len(parts) > 1:
                header += f" (part {i} of {len(parts)})"
            if breadcrumb:
                header += f"\n[{breadcrumb}]"
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{section}{suffix}",
                "doc_id": doc["doc_id"],
                "section": section,  # eval matches on doc_id#section, so all parts of a section count
                "part": i,
                "n_parts": len(parts),
                "text": f"{header}\n{body}",
                "body": body,
                "product_name": doc["product_name"],
                "category": doc["category"],
                "source_file": doc["source_file"],
                "pulled_on": doc["pulled_on"],
                "website_link": doc["website_link"],
                "stale_flags": stale_flags(f"{breadcrumb}\n{body}"),  # breadcrumb too: headings move there on split
            })
    return chunks


def build_chunks(docs=None, max_chars=MAX_CHUNK_CHARS):
    """Chunk every document. Raises if any chunk_id repeats, so citations stay unambiguous."""
    docs = build_documents() if docs is None else docs
    chunks = [c for d in docs for c in chunk_document(d, max_chars)]
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_ids")
    return chunks


# Small English stopword list: enough to stop "what is the ..." dominating BM25, without dropping banking words
# like "no", "not", "above", "below", which change meaning ("no initial deposit", "fall below fee").
STOPWORDS = set("""
a an the and or of to in on for with at by from as is are was were be been it its this that these those
what which who whom how when where why do does did can could should would will i my me we our you your
they their them he she his her there here any some about into than then so if
""".split())


def normalise_number(token):
    """Make numbers match however they're written: '3,000' -> '3000', '0.2500' -> '0.25', '1.20' -> '1.2'."""
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token


# Abbreviations used in the source data, mapped to the word staff would type.
ABBREVIATIONS = {"mth": "month", "mths": "month", "yr": "year", "yrs": "year", "pa": "annum"}


def light_stem(word):
    """Strip a plural 's' so 'accounts' matches 'account' ('fees' -> 'fee'). Leaves 'plus', 'class', 'basis' alone."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text):
    """Lowercase tokens for BM25: words, numbers (normalised), and the currency markers 's$' / 'us$'.

    '%' is dropped so '1.2%' matches '1.2'; '+' becomes 'plus' so 'Bonus+' matches 'bonus plus'.
    Hyphenated words split into parts ('fall-below' -> 'fall', 'below'). Plurals and the data's
    abbreviations ('mths') are normalised so they match how staff phrase questions.
    """
    text = text.lower().replace("+", " plus ")
    tokens = re.findall(r"us\$|s\$|\d+(?:[.,]\d+)*|[a-z]+", text)
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t[0].isdigit():
            out.append(normalise_number(t))
        else:
            out.append(ABBREVIATIONS.get(t) or light_stem(t))
    return out


# Query-side synonyms: everyday words staff or customers use -> the wording in OCBC's product data.
# Applied to the query only (the index is untouched), and every expansion is reported with the hit, so it stays
# explainable. Keep this list to general banking vocabulary, not fixes for individual eval questions.
# CAVEAT: entries marked (eval) also appear in the wording of data/eval/questions.json, so eval gains from them are
# optimistic; a held-out question set is needed to measure them fairly.
SYNONYMS = [
    (r"\bfixed deposits?\b|\bfds?\b", "time deposit"),
    (r"\b(set|fixed) term\b|\block (in|away)\b|\bguaranteed return", "time deposit tenor"),             # (eval) p01
    (r"\bmortgages?\b", "home loan"),                                                                # (eval) p05
    (r"\brefinanc\w*|\bswitch\w* (my |their |the )?(home )?loan", "refinancing"),
    (r"\bborrow\w*|\bhow much can (i|they|he|she|we) get\b", "loan amount"),                         # (eval) p05
    (r"\b(american|us) dollars?\b|\bgreenbacks?\b", "usd us$"),
    (r"\byen\b", "jpy"), (r"\beuros?\b", "eur"), (r"\b(british )?pounds?\b|\bsterling\b", "gbp"),      # (eval) p04
    (r"\b(renminbi|yuan|rmb)\b", "cnh"), (r"\b(australian|aussie) dollars?\b", "aud"),
    (r"\b(foreign|multiple|several|other) currenc\w*", "foreign currency currencies"),
    (r"\bpay ?(cheque|check)s?\b|\bpaychecks?\b|\bwages?\b|\bpay ?slips?\b", "salary"),              # (eval) p02
    (r"\b(youth|young (adult|people|person)s?|teen\w*|students?)\b", "youth age"),                   # (eval) p06
    (r"\b(kids?|child|children|son|daughter|\d+[- ]year[- ]old)\b", "children age"),                  # (eval) p08
    (r"\buntouched\b|\bnot (touch|withdraw)\w*|\bno withdrawals?\b|\bleav\w+ (it|their|my|the) money",
     "no withdrawals"),                                                                             # (eval) p03
    (r"\b(pay|earn|bear)s? (any )?interest\b|\binterest[- ]free\b", "interest bearing rate"),        # (eval) f06
    (r"\bcheque ?books?\b|\bcheck ?books?\b", "chequebook cheque"),
    (r"\bhow old\b|\bminimum age\b|\bage limit\b", "age eligibility"),
    (r"\bcharges?\b|\bcosts?\b", "fee"),
]


def expand_query(query):
    """Add data-side wording for any SYNONYMS pattern found in the query. Returns (expanded query, [added phrases])."""
    added = [extra for pattern, extra in SYNONYMS if re.search(pattern, query, re.IGNORECASE)]
    return (f"{query} {' '.join(added)}" if added else query), added


class BM25Retriever:
    """BM25 (Okapi) keyword search over chunks, with query-side synonym expansion. Each hit says which query terms
    it matched and which synonyms were added."""

    name = "bm25"

    def __init__(self, chunks, expand=True):
        self.chunks = chunks
        self.expand = expand
        self.chunk_tokens = [tokenize(c["text"]) for c in chunks]  # text includes the product/section header
        self.bm25 = BM25Okapi(self.chunk_tokens)
        if not expand:
            self.name = "bm25(no synonyms)"

    def search(self, query, k=8):
        """Return the top-k hits as dicts: {rank, score, chunk, matched_terms, expansions}. Zero-score chunks are dropped."""
        expanded, expansions = expand_query(query) if self.expand else (query, [])
        q_tokens = tokenize(expanded)
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            token_set = set(self.chunk_tokens[i])
            matched = sorted({t for t in q_tokens if t in token_set})
            hits.append({"rank": len(hits) + 1, "score": round(float(scores[i]), 3),
                         "chunk": self.chunks[i], "matched_terms": matched, "expansions": expansions})
        return hits


def print_hits(query, hits, label=""):
    """Print a query's hits in a compact, readable form."""
    print(f"\nQ: {label}{query}\n   tokens: {tokenize(query)}")
    for h in hits:
        flag = "  [stale]" if h["chunk"]["stale_flags"] else ""
        print(f"   {h['rank']}. {h['score']:>6.2f}  {h['chunk']['chunk_id']:<44} matched={h['matched_terms']}{flag}")
    if not hits:
        print("   (no hits)")


EMBED_MODEL = "minishlab/potion-base-8M"  # static embeddings: CPU-only, no torch (Intel Mac constraint)
EMBED_CACHE = "data/cache/embeddings.npz"
RRF_K = 60  # standard RRF constant; damps the gap between rank 1 and rank 2 so neither retriever dominates


def load_embedding_model(name=EMBED_MODEL):
    """Load the model2vec static embedding model (downloaded once to the Hugging Face cache)."""
    from model2vec import StaticModel  # imported lazily so BM25-only use doesn't pay the load cost
    return StaticModel.from_pretrained(name)


def _texts_key(texts, model_name):
    """Hash of the model name and chunk texts, so the cache is rebuilt whenever chunks change."""
    return hashlib.sha256((model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()


def embed_chunks(chunks, model, model_name=EMBED_MODEL, cache_path=EMBED_CACHE):
    """Return L2-normalised chunk vectors, from cache if the chunk texts and model are unchanged."""
    texts = [c["text"] for c in chunks]
    key = _texts_key(texts, model_name)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if str(cached["key"]) == key:
            return cached["vectors"]
    vectors = model.encode(texts)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, key=key, vectors=vectors)
    return vectors


class EmbeddingRetriever:
    """Semantic search: cosine similarity between the query and chunk vectors (model2vec)."""

    name = "embedding"

    def __init__(self, chunks, model=None):
        self.chunks = chunks
        self.model = model or load_embedding_model()
        self.vectors = embed_chunks(chunks, self.model)

    def similarities(self, query):
        """Cosine similarity of the query to every chunk (vectors are normalised, so a dot product)."""
        q = self.model.encode([query])[0]
        return self.vectors @ (q / np.linalg.norm(q))

    def search(self, query, k=8):
        """Return the top-k hits as dicts: {rank, score (cosine), chunk}."""
        sims = self.similarities(query)
        order = np.argsort(-sims)[:k]
        return [{"rank": r + 1, "score": round(float(sims[i]), 3), "chunk": self.chunks[i]}
                for r, i in enumerate(order)]


class HybridRetriever:
    """BM25 + embeddings fused with Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) over both rankings.

    RRF uses ranks, not raw scores, so BM25 scores (0-12) and cosines (0-1) never need rescaling. Each hit
    shows its BM25 rank, embedding rank and BM25 matched terms, so it's clear which retriever put it there.
    """

    name = "hybrid"

    def __init__(self, chunks, bm25=None, embedding=None, bm25_weight=1.0, embedding_weight=1.0):
        self.chunks = chunks
        self.bm25 = bm25 or BM25Retriever(chunks)
        self.embedding = embedding or EmbeddingRetriever(chunks)
        self.bm25_weight, self.embedding_weight = bm25_weight, embedding_weight  # weighted RRF; 1/1 = standard RRF
        if (bm25_weight, embedding_weight) != (1.0, 1.0):
            self.name = f"hybrid(bm25 x{bm25_weight:g}, emb x{embedding_weight:g})"

    def search(self, query, k=8):
        """Return the top-k fused hits: {rank, score (RRF), chunk, bm25_rank, embedding_rank, cosine, matched_terms}."""
        n = len(self.chunks)
        bm25_hits = {h["chunk"]["chunk_id"]: h for h in self.bm25.search(query, k=n)}
        emb_hits = {h["chunk"]["chunk_id"]: h for h in self.embedding.search(query, k=n)}
        fused = []
        for c in self.chunks:
            cid = c["chunk_id"]
            b, e = bm25_hits.get(cid), emb_hits[cid]
            score = self.embedding_weight / (RRF_K + e["rank"])
            score += self.bm25_weight / (RRF_K + b["rank"]) if b else 0  # no BM25 overlap -> no BM25 vote
            fused.append({"score": round(score, 5), "chunk": c, "bm25_rank": b["rank"] if b else None,
                          "embedding_rank": e["rank"], "cosine": e["score"],
                          "matched_terms": b["matched_terms"] if b else [],
                          "expansions": b["expansions"] if b else []})
        fused.sort(key=lambda h: -h["score"])
        return [dict(h, rank=r + 1) for r, h in enumerate(fused[:k])]


def describe_hit(hit):
    """One-line WHY for a hit, whichever retriever produced it."""
    syn = f", synonyms+{hit['expansions']}" if hit.get("expansions") else ""
    if "bm25_rank" in hit:
        return (f"bm25 #{hit['bm25_rank'] or '-'}, emb #{hit['embedding_rank']} (cos {hit['cosine']:.2f}), "
                f"matched={hit['matched_terms']}{syn}")
    if "matched_terms" in hit:
        return f"matched={hit['matched_terms']}{syn}"
    return f"cosine {hit['score']:.2f}"


def compare_retrievers(query, retrievers, k=3, expected=()):
    """Print each retriever's top-k for one query side by side, marking hits on an expected product with '*'."""
    print(f"\nQ: {query}" + (f"\n   expected product(s): {list(expected)}" if expected else ""))
    for r in retrievers:
        print(f"  {r.name}:")
        for h in r.search(query, k):
            mark = "*" if h["chunk"]["doc_id"] in expected else " "
            print(f"   {mark}{h['rank']}. {h['chunk']['chunk_id']:<42} {describe_hit(h)}")


# ---------------------------------------------------------------------------------------------------------------
# Step 5: answering
# ---------------------------------------------------------------------------------------------------------------

LLM_TIMEOUT_S = float(os.getenv("OCBC_QA_TIMEOUT", "20"))  # per attempt; a demo shouldn't hang on a slow call
LLM_MAX_RETRIES = 1  # one retry on timeouts/5xx/429, so the worst case is about 2 x LLM_TIMEOUT_S
FALLBACK_SECTIONS = 3  # retrieved sections shown when the AI answer is unavailable
DEFAULT_K = 8  # chunks sent to Claude; 5 was too few for questions spanning several products (step 7)
DEFAULT_MODEL = "claude-sonnet-5-5"  # override with OCBC_QA_MODEL; Sonnet is enough to read ~5 short chunks
MIN_BM25_SCORE = 2.0  # below this, no chunk shares meaningful words with the question: refuse without calling Claude
                      # (provisional; tuned on the eval set in step 7)

# ASSUMPTION (regulatory): this is a staff reference tool, not customer advice. The prompt forbids recommending
# products, in line with MAS fair-dealing expectations as I understand them; this has not been checked with compliance.
SYSTEM_PROMPT = """You answer questions from OCBC branch and contact-centre staff about OCBC products.

You will be given numbered source chunks taken from OCBC's public product data, then a question.

Rules:
- Answer ONLY from the chunks. Do not use outside knowledge about OCBC, banking, or regulation, even if you believe it is true.
- Put the chunk id in square brackets after every fact, e.g. "The minimum balance is S$3,000 [360-account#Minimum balance]."
- Set status:
  - "answered" when the chunks fully answer the question.
  - "partial" when they answer only part of it: answer that part and say in missing what is not covered.
  - "not_in_data" when they don't contain the answer at all. In answer, briefly say what the data does or doesn't show (with citations if you point to a chunk, e.g. a rate table with the needed row missing); say in missing what is absent. Do not guess.
- Never fill a gap in a rate table from a neighbouring row, tier or product. If the exact row is missing, say so.
- If chunks contradict each other, give both versions with their citations instead of choosing one.
- Rates and fees come from data pulled on a past date. Write "the data lists ..." rather than presenting any figure as current, and never say a rate "is available today".
- Do not recommend products or give financial advice. State what the data says; staff decide what to tell customers.
- Be concise: a short answer, then bullet points if there are several items."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "partial", "not_in_data"]},
        "answer": {"type": "string", "description": "Answer with [chunk_id] citations; for not_in_data, what the data does/doesn't show."},
        "citations": {"type": "array", "items": {"type": "string"}, "description": "chunk_ids actually used."},
        "missing": {"type": "string", "description": "What the chunks don't cover; empty if fully answered."},
    },
    "required": ["status", "answer", "citations", "missing"],
    "additionalProperties": False,
}

FRESHNESS_TEMPLATE = ("Source data pulled {pulled} from OCBC's public product API. Figures are indicative and may be "
                      "outdated; verify on ocbc.com before quoting rates or fees to a customer.")


def get_client():
    """Create an Anthropic client. The key comes from .env via dotenv and is never printed."""
    import anthropic
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    return anthropic.Anthropic()


def keyword_confidence(question, retriever):
    """Top BM25 score for the question: a cheap 'is there anything relevant at all?' signal for any retriever."""
    bm25 = retriever if isinstance(retriever, BM25Retriever) else retriever.bm25
    hits = bm25.search(question, k=1)
    return hits[0]["score"] if hits else 0.0


def format_chunks(hits):
    """Render retrieved chunks for the prompt, each tagged with its chunk_id and source file."""
    return "\n\n".join(
        f'<chunk id="{h["chunk"]["chunk_id"]}" source_file="{h["chunk"]["source_file"]}">\n{h["chunk"]["text"]}\n</chunk>'
        for h in hits)


def build_freshness_note(cited_chunks, all_chunks=()):
    """Build the source-and-freshness note in code, so it appears on every answer whatever the model writes.

    Lists each cited product with its source file and link, plus any staleness flags on the cited chunks.
    """
    pulled = sorted({c["pulled_on"] for c in (cited_chunks or all_chunks)} or {"(unknown date)"})
    lines = [FRESHNESS_TEMPLATE.format(pulled=", ".join(pulled))]
    by_product = {}
    for c in cited_chunks:
        p = by_product.setdefault(c["product_name"], {"source_file": c["source_file"], "link": c["website_link"], "flags": []})
        p["flags"] += [f for f in c["stale_flags"] if f not in p["flags"]]
    for name, p in by_product.items():
        lines.append(f"- {name}: {p['source_file']} ({p['link']})")
        lines += [f"  ⚠ {f}" for f in p["flags"]]
    return "\n".join(lines)


def _refusal(question, hits, reason, chunks, model=None, usage=None, explanation="", cited=()):
    """Assemble a not_in_data result. Any explanation the model gave (e.g. 'that table row is missing') is kept."""
    cited = list(cited)
    return {"question": question, "status": "not_in_data", "refused": True, "answer": explanation,
            "refusal_reason": reason, "citations": _citation_list(cited), "dropped_citations": [],
            "freshness_note": build_freshness_note(cited, [h["chunk"] for h in hits] or chunks),
            "hits": hits, "model": model, "usage": usage}


def _citation_list(chunks):
    """Compact citation records for display and evaluation."""
    return [{"chunk_id": c["chunk_id"], "product_name": c["product_name"], "section": c["section"],
             "source_file": c["source_file"]} for c in chunks]


def _validate_citations(result, hits):
    """Keep only cited ids (declared or inline [..#..]) that were actually retrieved. Returns (valid chunks, dropped ids)."""
    retrieved = {h["chunk"]["chunk_id"]: h["chunk"] for h in hits}
    inline = re.findall(r"\[([^\[\]]+#[^\[\]]+)\]", result["answer"])
    cited_ids = list(dict.fromkeys(result["citations"] + inline))  # declared + inline, de-duplicated, in order
    return [retrieved[c] for c in cited_ids if c in retrieved], [c for c in cited_ids if c not in retrieved]


def call_claude(client, question, hits, model):
    """Ask Claude for a JSON answer grounded in the hits. Returns (parsed dict or None, stop_reason, usage)."""
    response = client.with_options(timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES).beta.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<chunks>\n{format_chunks(hits)}\n</chunks>\n\nQuestion: {question}"}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model in the same call
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason in ("refusal", "max_tokens"):
        return None, response.stop_reason, usage
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.stop_reason, usage


def _ai_unavailable(question, hits, chunks, reason, model=None, usage=None):
    """Fallback when Claude fails or is too slow: show the top retrieved sections with their citations instead.

    Retrieval needs no network, so staff still get the most relevant product data to read themselves.
    """
    top = [h["chunk"] for h in hits[:FALLBACK_SECTIONS]]
    return {"question": question, "status": "ai_unavailable", "refused": False, "answer": "",
            "refusal_reason": "AI answer unavailable right now. Showing the most relevant product sections instead; "
                              "read them directly.",
            "error": reason, "citations": _citation_list(top), "dropped_citations": [],
            "fallback_sections": [{"chunk_id": c["chunk_id"], "product_name": c["product_name"],
                                   "section": c["section"], "source_file": c["source_file"], "text": c["body"]}
                                  for c in top],
            "freshness_note": build_freshness_note(top, chunks), "hits": hits, "model": model, "usage": usage}


def answer(question, retriever, k=DEFAULT_K, client=None, model=None, min_bm25_score=MIN_BM25_SCORE):
    """Answer a staff question from retrieved chunks only.

    Returns {question, status (answered | partial | not_in_data | ai_unavailable), refused (= status == not_in_data), answer,
    refusal_reason (what's missing), citations[{chunk_id, product_name, section, source_file}], dropped_citations,
    freshness_note, hits, model, usage}. Citations are post-checked: any id not in the retrieved set is dropped,
    and an answer left with no valid citation is turned into not_in_data. If the Claude call fails, times out or
    returns something unusable, the result is ai_unavailable with the top retrieved sections (never an exception).
    """
    model = model or os.getenv("OCBC_QA_MODEL", DEFAULT_MODEL)
    hits = retriever.search(question, k=k)
    chunks = retriever.chunks
    if not hits or keyword_confidence(question, retriever) < min_bm25_score:
        return _refusal(question, hits, "No product data matches this question closely enough.", chunks)  # no LLM call

    try:
        result, stop_reason, usage = call_claude(client or get_client(), question, hits, model)
        if result is not None:
            result = {"status": result["status"], "answer": result["answer"], "citations": list(result["citations"]),
                      "missing": result["missing"]}  # KeyError here = malformed output -> fallback
    # Deliberately broad: timeouts, network/auth/rate-limit errors, bad JSON or a missing key must never crash the
    # demo. The error type is kept on the result (never the message, which could echo request details).
    except Exception as e:  # noqa: BLE001
        return _ai_unavailable(question, hits, chunks, type(e).__name__, model)
    if result is None:  # safety refusal or max_tokens: no usable answer
        return _ai_unavailable(question, hits, chunks, f"stop_reason={stop_reason}", model, usage)
    cited, dropped = _validate_citations(result, hits)
    if result["status"] == "not_in_data":
        return _refusal(question, hits, result["missing"] or "Not covered by the product data.", chunks, model, usage,
                        explanation=result["answer"], cited=cited)
    if not cited:
        return _refusal(question, hits, "Answer had no citation to the retrieved product data, so it was withheld.",
                        chunks, model, usage)
    return {
        "question": question,
        "status": result["status"],  # answered | partial
        "refused": False,
        "answer": result["answer"],
        "refusal_reason": result["missing"],  # for partial: the part of the question that isn't covered
        "citations": _citation_list(cited),
        "dropped_citations": dropped,
        "freshness_note": build_freshness_note(cited, chunks),
        "hits": hits,
        "model": model,
        "usage": usage,
    }


def print_answer(result):
    """Print an answer result for the terminal."""
    print(f"\nQ: {result['question']}")
    print("   retrieved: " + ", ".join(f"{h['chunk']['chunk_id']} ({h['score']})" for h in result["hits"]))
    print(f"   STATUS: {result['status']}" + (f"  (missing: {result['refusal_reason']})" if result["refusal_reason"] else ""))
    if result["answer"]:
        print("   " + result["answer"].replace("\n", "\n   "))
    for sec in result.get("fallback_sections", []):
        print(f"   --- {sec['product_name']} / {sec['section']} [{sec['chunk_id']}]\n   " + sec["text"][:300].replace("\n", "\n   "))
    if result.get("error"):
        print(f"   (error: {result['error']})")
    if result["citations"]:
        print("   citations: " + "; ".join(f"{c['product_name']} [{c['chunk_id']}] ({c['source_file']})" for c in result["citations"]))
    if result["dropped_citations"]:
        print(f"   dropped invalid citations: {result['dropped_citations']}")
    print("   " + result["freshness_note"].replace("\n", "\n   "))
    if result["usage"]:
        print(f"   [{result['model']}, {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out tokens]")


if __name__ == "__main__":
    import sys

    chunks = build_chunks()
    retriever = BM25Retriever(chunks)
    client = get_client()
    demo = [
        "What promotional rate does a S$30,000 SGD time deposit for 12 months earn?",  # n01 answerable, tiered
        "What is the SIBOR-based home loan rate?",                                      # s01 stale
        "What is the time deposit rate for S$300,000 placed for 4 months?",            # n09 missing table row
        "What is the annual fee for the OCBC 365 Credit Card?",                        # u01 not in data
    ]
    for q in sys.argv[1:] or demo:
        print_answer(answer(q, retriever, client=client))
