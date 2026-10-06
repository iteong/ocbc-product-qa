# Handoff: OCBC Product Q&A assistant (prototype)

Staff-facing Q&A over OCBC product data. It answers only from retrieved product sections, cites `product_name` + `source_file`, refuses when the data doesn't cover the question, and adds a source-and-freshness note to every answer.

## Run it
Python 3.11. Run `source ~/Source/ocbc-env/.venv/bin/activate`, then put `ANTHROPIC_API_KEY` in `.env`. Run everything from the repo root:
- `streamlit run app.py`: the app (Ask + Evaluation tabs).
- `python -m src.core "question"`: answer one question in the terminal.
- `python -m src.eval`: BM25 vs hybrid evaluation, about 88 Claude calls (~$0.70). Add `--retrieval-only` for no API calls.
- `python scripts/clean_products.py`, then `python -m src.data_gen`: rebuild clean data and product docs.

## Files
| File | What it does |
|---|---|
| `scripts/clean_products.py` | Raw API JSON → one schema in `data/products_clean/` (+ `DATA_QUALITY.md`). |
| `src/data_gen.py` | One document per product (`data/docs/*.md` for reading). |
| `src/core.py` | Chunking and stale flags, BM25 + synonyms, model2vec embeddings, hybrid (RRF), Claude answering (citation checks, timeout, AI-unavailable fallback). |
| `src/eval.py` | Question-set validation, retrieval and answer metrics → `data/eval/SUMMARY.md` + CSVs. |
| `data/eval/questions.json` | 44 hand-written questions (6 types, including should-refuse and stale traps). `baseline_v1/` holds the pre-tuning results. |
| `app.py` | Streamlit UI. Never shows a stack trace: on API error or timeout it shows the top 3 sections, labelled "AI answer unavailable". |

## Data source and what should replace it
Product facts come from **OCBC's public API (Connect2OCBC)**, pulled **5 Oct 2026**, which marks them *indicative and subject to change*. `scripts/clean_products.py` cleans them. There are 10 products, with no credit cards, live FD rates or promotions. **Replace with the bank's governed internal source of truth**: the product master / rates system of record, plus approved product disclosure documents (product highlights sheets, T&Cs, fee schedules). These need effective dates and named owners. Keep the citation and freshness contract when switching.

## Data quality
See [`data/products_clean/DATA_QUALITY.md`](data/products_clean/DATA_QUALITY.md):
- the TD rate table is missing the 3–5 month row for S$250k–S$499k and has duplicate rows;
- SIBOR-based home loan packages (SIBOR is discontinued);
- unit trust returns "as of March 2016" (back-tested);
- `Nov16` campaign links;
- empty `key_risks`;
- conflicting FRANK age wording.

## Key assumptions
- An internal staff reference, not customer advice. The prompt forbids product recommendations. This hasn't been reviewed by compliance against MAS fair-dealing expectations.
- Every figure is treated as possibly outdated. Stale flags (SIBOR, promotional, "as of", back-tested, Nov16) are regex rules I wrote, not OCBC metadata.
- No customer data is used anywhere.

## Known shortcuts
- **Eval is optimistic.** The 44 questions are also how synonyms and k=8 were tuned. 7 of 19 synonyms match eval wording (marked `(eval)`), so BM25's 44/44 needs a **held-out question set** written by someone else.
- **Fixed settings.** Synonyms are a hand list. The BM25 refusal threshold (2.0) never fires, so Claude does the refusing. Hybrid uses an untuned RRF and loses rate tables, so BM25 is the default.
- **Crude grading.** Grading is substring matching, with no LLM-judge or human review. There's one model (`claude-sonnet-5-5`, via `OCBC_QA_MODEL`), single-turn only, and no tests beyond the eval.

## To productionise (data science squad)
1. **Data:** ingest from the source of truth on a schedule, with effective dates, change detection and data-quality checks that block bad loads.
2. **Eval:** a held-out set and real staff questions from logs. Human review of correctness and refusals, plus regression gates in CI.
3. **Retrieval:** structured lookup for rate tables (amount × tenor) instead of text search. Learn synonyms from query logs and re-test hybrid or rerankers on held-out data.
4. **Governance:** compliance sign-off on prompt and refusal policy; model risk review; audit log of question, sources and answer; PII redaction on input.
5. **Ops:** auth/SSO, rate limits, monitoring of refusal/fallback rates and cost, and a feedback button wired into the eval set.
