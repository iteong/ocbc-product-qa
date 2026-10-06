# Session log: building the OCBC product Q&A assistant

A record of one Claude Code session (6 Oct 2026): what was asked, what was built, the decisions taken and why, results, problems found, and open caveats. For how to run the project, see [`HANDOFF.md`](../HANDOFF.md).

## Starting point
- **Repo:** `ocbc-interview`, renamed `ocbc-product-qa` at the end. It held cleaned OCBC product data (`data/products_clean/products.json`, 10 products, pulled 5 Oct 2026 from OCBC's public API), a data-quality report, and empty stubs for `src/data_gen.py`, `src/core.py`, `src/eval.py` and `app.py`.
- **Project rules** (`CLAUDE.md`):
  - Python 3.11; pandas, scikit-learn, Streamlit; minimal dependencies;
  - simple, explainable methods first, and every ranking must show WHY;
  - cite `product_name` and `source_file`, and never present rates as current;
  - flag banking and regulatory assumptions;
  - model2vec rather than torch (Intel Mac);
  - dotenv for secrets;
  - run code after each change and show output;
  - don't run `streamlit run`.
- **Working style you asked for:** plan first with no code, then one step per prompt, waiting for approval each time.

## 1. Plan (approved)
Eight steps:
1. one document per product;
2. an evaluation question set;
3. section-level chunking;
4. BM25 retrieval;
5. Claude answering only from retrieved chunks, with citations, refusals and a source-and-freshness note;
6. hybrid retrieval with local embeddings, if time allowed;
7. evaluation of BM25 vs hybrid;
8. a Streamlit app.

## 2. Step 1: product documents (`src/data_gen.py`)
- **What I built:** each product becomes a dict of metadata plus its non-empty sections, in a fixed order, with readable titles. Each is also written to `data/docs/<doc_id>.md` for people to read. Ids are filename-safe (`Bonus+` → `bonus-plus`), and a duplicate id is an error.
- **Why the disclaimer is metadata, not a section:** it's identical boilerplate on every product, so as a section it would only add noise to search. It's kept for the freshness note.
- **Result:** 10 documents, 57 sections.
- **Found in the data:**
  - Time Deposit has no description;
  - Home Loan Refinancing has no eligibility section;
  - FRANK's age wording conflicts (16–26 vs 16+).

## 3. Step 2: evaluation questions (`data/eval/questions.json`)
- **The set:** 44 hand-written questions in six types:
  - factual (8);
  - paraphrase (8), with no product words in the question;
  - numeric/rate tiers (9);
  - cross-product (6);
  - unanswerable (8);
  - stale traps (5).
- **Fields:** expected products and sections, required strings (`must_contain`), whether it should refuse, and whether it should carry a stale flag.
- **Traps built from the data's own gaps:**
  - a TD rate row that doesn't exist;
  - refinancing eligibility, which has no section;
  - unit trust risks, where `key_risks` is empty;
  - credit cards, SORA and CPF, none of which are in the data.
- **Validation** (`validate_questions` in `src/eval.py`) checks that every expected section exists, and that every required string actually appears in an expected section. This stops the eval rewarding facts that aren't in the source.

## 4. Step 3: chunking (`src/core.py`)
- **Granularity:** one chunk per section. Each chunk's text starts with a header, `"{product} ({category}) — {section}"`, because short sections like `S$0` mean nothing on their own.
- **Splitting long sections:** sections over about 900 characters are split along their own structure, never mid-block. A heading stays with its bullets, and a bullet stays with its children. When a block is still too big, its heading becomes a breadcrumb on each part, so no TD rate tier is ever split. Result: 57 sections → 63 chunks, 6 of them split.
- **Stale flags:** regex rules for SIBOR, "as of" a past date, back-tested returns, `Nov16` links, and promotional rates. These are my assumptions, commented as such in the code.
- **Bug fixed:** the "Promotional" heading moved into the breadcrumb on split TD parts, so those parts lost their flag. Flags are now computed on breadcrumb plus body.

## 5. Step 4: BM25 (`src/core.py`)
- **Tokenizer:**
  - lowercases; numbers are normalised (`3,000` → `3000`, `0.2500` → `0.25`) and `%` is dropped;
  - `S$` and `US$` are kept as tokens, and `+` becomes `plus`;
  - plural "s" is stripped, and the data's abbreviations are expanded (`mths` → `month`);
  - the stopword list is short and keeps `no`, `not` and `below`.
- **WHY:** each hit lists the query terms it matched.
- **Bugs fixed:** my display label was being tokenised into the query, and "mths" didn't match "months".
- **Finding:** paraphrases fail (4/8). The question "credit card fee" scores as high as real answers, so a BM25 score threshold can't do the refusing; Claude has to.

## 6. Embeddings and hybrid (step 6, brought forward at your request)
- **What I built:**
  - `EmbeddingRetriever`: model2vec `potion-base-8M`, with cached vectors that rebuild when chunk text changes;
  - `HybridRetriever`: Reciprocal Rank Fusion, later with weights;
  - `compare_retrievers()`.
- **Your two test questions:**
  - "Young adults, no minimum balance": all three retrievers found FRANK.
  - "Savings in American dollars": BM25 found USD Current Account at #2. The embedding and hybrid retrievers ranked short "Savings Account" chunks first, with the first relevant chunk at #8.
- **Diagnosis:** potion-base-8M averages word vectors, so the header dominates short chunks. Embedding only the body fixed that one question but was worse overall, so the headers stayed.
- **Lesson:** embeddings are strong on paraphrase (7/8) but weak on numbers (3/8), and equal-weight fusion averaged those strengths away.

## 7. Step 5: Claude answering (`src/core.py`)
- **Model:** `claude-sonnet-5-5`, per the approved plan; change it with `OCBC_QA_MODEL`.
- **Output:** JSON via a fixed schema.
- **Prompt rules:**
  - answer only from the chunks, and cite `[chunk_id]` after every fact;
  - never fill a gap in a rate table from a neighbouring row;
  - show both versions when sources conflict;
  - write "the data lists…", never "available today";
  - no product recommendations. This is my assumption about fair dealing for a staff reference tool and hasn't been checked with compliance.
- **Code-side safeguards:**
  - citations not in the retrieved set are dropped, and an answer with no valid citation becomes a refusal;
  - the freshness note is built in code (pull date, verify on ocbc.com, source file and link per product, stale flags), so it always appears;
  - server-side refusal fallbacks are on.
- **Your request:** a `status` field: `answered`, `partial` or `not_in_data`. This came after the missing-row question returned an honest "not listed" answer that the eval scored as a non-refusal. A `not_in_data` reply keeps its cited explanation.

## 8. Step 7: evaluation (`src/eval.py`)
- **Retrieval metrics, no LLM:** section hit@1/3/5/8, MRR, product hit, section recall. Also a sweep of hybrid weights and a check of the BM25 refusal threshold.
- **Answer metrics:** overall correct, required facts present, expected section cited, refusal precision and recall, stale warning present, dropped citations, AI-unavailable count, tokens and cost.
- **Outputs:** CSVs and `data/eval/SUMMARY.md`.
- **First run (baseline, saved in `data/eval/baseline_v1/`):**

  | | Retrieval: correct section in top 3 | Answers: overall correct | Should-refuse questions refused |
  |---|---|---|---|
  | BM25 | 0.86 | 0.82 | 0.89 |
  | Hybrid | 0.77 | 0.80 | 0.78 |

  Every failure traced back to retrieval. Claude never invented a fact, and no citation needed dropping. The refusal threshold (2.0) never fires: should-refuse questions score from 4.04, answerable ones from 4.66.
- **My recommendation:** improve retrieval, not the prompt. Raise k from 5 to 8 and add a synonym list.

## 9. Applying the recommendation
- **What changed:**
  - 19 query-side synonym rules (fixed deposit → time deposit, mortgage → home loan, American dollars → USD, …), each reported with the hit;
  - `DEFAULT_K = 8`.
- **Results:**

  | | Before → after |
  |---|---|
  | BM25 retrieval, correct section in top 8 | 0.89 → **1.00** |
  | BM25 answers, overall correct | 0.82 → **1.00** (44/44) |
  | Hybrid answers, overall correct | 0.80 → 0.89 |

  Cost rose about 20%, to $0.34 per 44-question run.
- **What improved:**
  - synonyms fixed f06 and p01, p03, p04, p05;
  - k=8 fixed c02 and c04.
- **One regression:** s05 dropped from rank 2 to 5, still within the 8 chunks sent.
- **Caveat recorded:** 7 of the 19 synonyms match eval wording (marked `(eval)`), and k was chosen on these same questions. So 44/44 is a development-set score; a held-out set written by someone else is needed.
- **Decision:** BM25 with synonyms is the default; hybrid is a toggle. Hybrid's remaining failures are all TD rate tables.

## 10. Step 8: Streamlit app (`app.py`)
- **Layout:** sidebar with the retriever toggle, k, the model and a data notice. The Ask tab has examples, a status banner, the answer with inline citations, what's missing, cited sources, the freshness box, and a "Why these sources?" expander. The Evaluation tab shows the summary and per-question results.
- **Testing:** headless with Streamlit `AppTest`; I didn't run `streamlit run`. Deprecated `use_container_width` was replaced with `width="stretch"`.
- **"streamlit: command not found":** your new tab didn't have the virtual environment activated. Fix: `source ~/Source/ocbc-env/.venv/bin/activate`.

## 11. Robustness and handoff (your request before committing)
- **Timeout:** Claude calls time out after 20 seconds (`OCBC_QA_TIMEOUT`) with one retry.
- **Fallback:** any failure returns `ai_unavailable` with the top 3 retrieved sections and their citations, labelled "AI answer unavailable", instead of crashing. Failures covered: timeout, network, auth, rate limit, malformed output, a safety stop, or the client failing to start.
- **Error details:** only the error type is kept, never the message.
- **App behaviour:** fallbacks aren't cached, so asking again retries. Hybrid falls back to BM25 if it can't load. An outer catch shows a plain error for anything unexpected.
- **Tested:** normal, timeout, invalid key and no client, each in its own process. An earlier `AttributeError` came from my test harness reloading modules, not from the app.
- **`HANDOFF.md`:** how to run, files, data provenance and what should replace it, data quality, assumptions, shortcuts, and steps to productionise.
- **`.gitignore`:** now excludes `data/cache/`.

## 12. Commit and rename
- **Commit:** `6009224` on `main`, as you asked. 25 files; a scan of the staged changes found no secrets. Not pushed, and no remote is set.
- **Rename:** before renaming, I checked that nothing referenced the folder name. I stopped the running Streamlit app, renamed the folder `~/Source/ocbc-interview` → `~/Source/ocbc-product-qa`, and checked that git and the code still worked.

## Open items
1. **Held-out questions** (10–15, written by someone who hasn't seen the synonyms), to measure the synonyms and k fairly.
2. **Compliance review** of the prompt and refusal policy against MAS fair-dealing expectations.
3. **Undecided points:**
   - flag every home loan rate section as indicative (s03 relies on the SIBOR chunk);
   - extend stale rules to `Nov16` links in product URLs;
   - change refusal wording from "the chunks" to "the product data".
4. **Replace the public API data** with the bank's governed product and rates source, and use structured lookup for rate tables. See `HANDOFF.md`.
5. **Push** to a remote once it's created.
