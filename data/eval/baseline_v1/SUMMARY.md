# Evaluation: BM25 vs hybrid

Product data pulled 2026-10-05 (indicative). Questions: data/eval/questions.json. Small hand-written set (44 questions): treat differences of 1-2 questions as noise.

## Retrieval (35 answerable questions, no LLM)

| retriever | hit@1 | hit@3 | hit@5 | MRR | product_hit@5 | section_recall@5 |
|---|---|---|---|---|---|---|
| bm25 | 0.69 | 0.86 | 0.86 | 0.77 | 0.97 | 0.81 |
| hybrid | 0.54 | 0.77 | 0.86 | 0.66 | 0.94 | 0.81 |
| embedding | 0.51 | 0.74 | 0.80 | 0.64 | 0.91 | 0.75 |

### Section hit@3 by question type

| type | bm25 | hybrid | embedding |
|---|---|---|---|
| factual | 0.88 | 1.00 | 0.88 |
| paraphrase | 0.50 | 0.62 | 0.88 |
| numeric | 1.00 | 0.62 | 0.38 |
| cross_product | 1.00 | 1.00 | 1.00 |
| stale | 1.00 | 0.60 | 0.60 |

### Hybrid weight sensitivity (hit@3 / MRR)

Tuned on the same questions it is scored on, so this shows sensitivity, not a validated setting.

|  | hit@3 | MRR | paraphrase hit@3 | numeric hit@3 |
|---|---|---|---|---|
| bm25 x1, emb x1 | 0.77 | 0.66 | 0.62 | 0.62 |
| bm25 x2, emb x1 | 0.74 | 0.71 | 0.50 | 0.75 |
| bm25 x3, emb x1 | 0.80 | 0.72 | 0.50 | 0.88 |
| bm25 x1, emb x2 | 0.74 | 0.62 | 0.62 | 0.50 |
| bm25 x1, emb x3 | 0.77 | 0.65 | 0.75 | 0.50 |

### Pre-LLM refusal threshold (top BM25 score)

| should_refuse | count | min | 50% | max |
|---|---|---|---|---|
| answerable | 35.00 | 4.66 | 9.81 | 28.05 |
| should refuse | 9.00 | 4.04 | 7.24 | 13.92 |

## Answers (claude-sonnet-5-5, all 44 questions)

|  | bm25 | hybrid |
|---|---|---|
| overall correct | 0.82 | 0.80 |
| facts ok (answerable) | 0.80 | 0.80 |
| cited expected section | 0.86 | 0.86 |
| refusal recall | 0.89 | 0.78 |
| refusal precision | 0.62 | 0.50 |
| false refusals | 5.00 | 7.00 |
| stale warning ok | 0.80 | 0.80 |
| partial answers | 6.00 | 9.00 |
| dropped citations | 0.00 | 0.00 |
| LLM calls | 44.00 | 44.00 |
| est. cost USD | 0.29 | 0.28 |

### Overall correct by question type

| type | bm25 | hybrid |
|---|---|---|
| factual | 0.88 | 1.00 |
| paraphrase | 0.50 | 0.62 |
| numeric | 1.00 | 0.67 |
| cross_product | 0.67 | 1.00 |
| unanswerable | 0.88 | 0.75 |
| stale | 1.00 | 0.80 |

### Wrong answers

- **bm25 / f06** (factual, status=not_in_data): wrong refusal decision. Retrieved: ['usd-current-account#Description', '360-account#Benefits', 'usd-current-account#Fees and charges']
- **bm25 / p01** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['frank-account#How to apply', 'unit-trusts#Overview', 'monthly-savings-account#Description']
- **bm25 / p03** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['360-account#Remarks', 'unit-trusts#Description', 'frank-account#Benefits']
- **bm25 / p04** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['monthly-savings-account#Eligibility', 'monthly-savings-account#How to apply', '360-account#How to apply-2']
- **bm25 / p05** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['home-loan-refinancing#Description', 'usd-current-account#Benefits', 'usd-current-account#Initial deposit']
- **bm25 / c02** (cross_product, status=answered): facts 1/2. Retrieved: ['frank-account#Eligibility', 'frank-account#Initial deposit', '360-account#How to apply-2']
- **bm25 / c04** (cross_product, status=partial): facts 2/3. Retrieved: ['frank-account#Eligibility', 'monthly-savings-account#Eligibility', '360-account#Eligibility']
- **bm25 / u06** (unanswerable, status=partial): wrong refusal decision. Retrieved: ['home-loan-refinancing#Description', 'home-loan-new-purchase#Eligibility', 'home-loan-refinancing#Interest rates-2']
- **hybrid / p01** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['bonus-plus-savings-account#Description', 'monthly-savings-account#Description', 'monthly-savings-account#Benefits-1']
- **hybrid / p03** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['360-account#Remarks', '360-account#Initial deposit', '360-account#How to apply-1']
- **hybrid / p05** (paraphrase, status=not_in_data): wrong refusal decision. Retrieved: ['home-loan-new-purchase#Description', 'home-loan-refinancing#Description', 'usd-current-account#Benefits']
- **hybrid / n01** (numeric, status=not_in_data): wrong refusal decision. Retrieved: ['360-account#Benefits', 'monthly-savings-account#Benefits-1', 'bonus-plus-savings-account#Description']
- **hybrid / n02** (numeric, status=not_in_data): wrong refusal decision. Retrieved: ['home-loan-refinancing#Interest rates-1', 'home-loan-new-purchase#Interest rates-1', 'time-deposit-fixed-deposit#Benefits-2']
- **hybrid / n03** (numeric, status=not_in_data): wrong refusal decision. Retrieved: ['360-account#Description', 'time-deposit-fixed-deposit#Remarks', '360-account#Benefits']
- **hybrid / u06** (unanswerable, status=partial): wrong refusal decision. Retrieved: ['home-loan-refinancing#Description', 'home-loan-refinancing#Loan amount', 'home-loan-refinancing#Interest rates-2']
- **hybrid / u08** (unanswerable, status=partial): wrong refusal decision. Retrieved: ['unit-trusts#How to apply', 'unit-trusts#Description', 'unit-trusts#How it works']
- **hybrid / s04** (stale, status=not_in_data): wrong refusal decision. Retrieved: ['home-loan-new-purchase#Interest rates-1', 'home-loan-refinancing#Interest rates-1', 'bonus-plus-savings-account#Initial deposit']
