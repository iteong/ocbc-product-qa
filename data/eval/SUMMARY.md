# Evaluation: BM25 vs hybrid

Product data pulled 2026-10-05 (indicative). Questions: data/eval/questions.json. Small hand-written set (44 questions): treat differences of 1-2 questions as noise.

## Retrieval (35 answerable questions, no LLM)

| retriever | hit@1 | hit@3 | hit@5 | hit@8 | MRR | product_hit@8 | section_recall@8 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.77 | 0.91 | 0.97 | 1.00 | 0.85 | 1.00 | 0.98 |
| hybrid | 0.60 | 0.83 | 0.89 | 0.94 | 0.72 | 1.00 | 0.93 |
| embedding | 0.51 | 0.74 | 0.80 | 0.89 | 0.65 | 0.97 | 0.85 |
| bm25(no synonyms) | 0.69 | 0.86 | 0.86 | 0.89 | 0.77 | 0.97 | 0.87 |

### Section hit@3 by question type

| type | bm25 | hybrid | embedding | bm25(no synonyms) |
|---|---|---|---|---|
| factual | 1.00 | 1.00 | 0.88 | 0.88 |
| paraphrase | 0.75 | 0.88 | 0.88 | 0.50 |
| numeric | 1.00 | 0.62 | 0.38 | 1.00 |
| cross_product | 1.00 | 1.00 | 1.00 | 1.00 |
| stale | 0.80 | 0.60 | 0.60 | 1.00 |

### Hybrid weight sensitivity (hit@3 / MRR)

Tuned on the same questions it is scored on, so this shows sensitivity, not a validated setting.

|  | hit@3 | MRR | paraphrase hit@3 | numeric hit@3 |
|---|---|---|---|---|
| bm25 x1, emb x1 | 0.83 | 0.72 | 0.88 | 0.62 |
| bm25 x2, emb x1 | 0.86 | 0.78 | 0.88 | 0.75 |
| bm25 x3, emb x1 | 0.86 | 0.79 | 0.75 | 0.88 |
| bm25 x1, emb x2 | 0.80 | 0.68 | 0.88 | 0.50 |
| bm25 x1, emb x3 | 0.80 | 0.68 | 0.88 | 0.50 |

### Pre-LLM refusal threshold (top BM25 score)

| should_refuse | count | min | 50% | max |
|---|---|---|---|---|
| answerable | 35.00 | 5.59 | 11.08 | 28.05 |
| should refuse | 9.00 | 4.04 | 8.47 | 13.92 |

## Answers (claude-sonnet-5-5, all 44 questions)

|  | bm25 | hybrid |
|---|---|---|
| overall correct | 1.00 | 0.89 |
| facts ok (answerable) | 1.00 | 0.89 |
| cited expected section | 1.00 | 0.94 |
| refusal recall | 1.00 | 0.89 |
| refusal precision | 1.00 | 0.67 |
| false refusals | 0.00 | 4.00 |
| stale warning ok | 0.80 | 0.80 |
| partial answers | 8.00 | 9.00 |
| dropped citations | 0.00 | 0.00 |
| LLM calls | 44.00 | 44.00 |
| est. cost USD | 0.34 | 0.34 |

### Overall correct by question type

| type | bm25 | hybrid |
|---|---|---|
| factual | 1.00 | 1.00 |
| paraphrase | 1.00 | 1.00 |
| numeric | 1.00 | 0.67 |
| cross_product | 1.00 | 1.00 |
| unanswerable | 1.00 | 0.88 |
| stale | 1.00 | 0.80 |

### Wrong answers

- **hybrid / n01** (numeric, status=not_in_data): wrong refusal decision. Retrieved: ['360-account#Benefits', 'monthly-savings-account#Benefits-1', 'bonus-plus-savings-account#Description']
- **hybrid / n02** (numeric, status=not_in_data): wrong refusal decision. Retrieved: ['home-loan-refinancing#Interest rates-1', 'home-loan-new-purchase#Interest rates-1', 'time-deposit-fixed-deposit#Benefits-2']
- **hybrid / n03** (numeric, status=not_in_data): wrong refusal decision. Retrieved: ['360-account#Description', 'time-deposit-fixed-deposit#Remarks', '360-account#Benefits']
- **hybrid / u06** (unanswerable, status=partial): wrong refusal decision. Retrieved: ['home-loan-refinancing#Description', 'home-loan-refinancing#Loan amount', 'home-loan-refinancing#Interest rates-2']
- **hybrid / s04** (stale, status=not_in_data): wrong refusal decision. Retrieved: ['home-loan-new-purchase#Interest rates-1', 'home-loan-refinancing#Interest rates-1', 'time-deposit-fixed-deposit#Remarks']
