# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected parts, all on synthetic and fictional data (the brand name only frames the scenario). All amounts are in INR. Every language-model call goes through the supplied offline `mock_llm.py`; no network, API key or paid service is used anywhere.

| Part | What it does | Code |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, answers 7 SQL questions, cross-checks in pandas | `src/part_a_*.py`, `sql/queries.sql` |
| B | RFM segments (K-means), lapse prediction model, retention list | `src/part_b_segments.py`, `src/part_b_lapse.py` |
| C | Ticket triage: guardrails, PII redaction, retries, validation, token budget, routing | `src/part_c_triage.py` |
| D | Policy RAG assistant and a resumable refund agent | `src/part_d_rag.py`, `src/part_d_agent.py` |

`generate_data.py` and `mock_llm.py` are the supplied files, unmodified. Libraries: standard library, pandas, numpy, scikit-learn, joblib (versions pinned in `requirements.txt`; Python 3.11+).

## Reproduce

Run from the repository root. Each script reads the outputs of the earlier ones, so keep the order.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a_clean.py` (audit, cleaning, CSVs and `data/instamart.db`)
6. `python src/part_a_validate.py` (prints `ALL CHECKPOINTS PASSED`)
7. `python src/part_a_sql_pandas.py` (runs `sql/queries.sql`, pandas cross-check, statistics)
8. `python src/part_b_segments.py` (RFM, K-means, PCA, `outputs/customer_segments.csv`)
9. `python src/part_b_lapse.py` (lapse model, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`)
10. `python src/part_c_triage.py` (`outputs/triage_budget_demo.csv`, `outputs/triage_results.csv`)
11. `python src/part_d_rag.py` (chunking, retrieval, evaluation)
12. `python src/part_d_agent.py` (`outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/`)

`./run_all.sh` runs steps 4 to 12 in one go once the environment is active. The full pipeline was replayed from an empty `data/`, `outputs/` and `state/` folder and every checkpoint below was reproduced.

## Part A

### Audit of the raw data

| Table | Shape | Missing values (count, %) |
|---|---|---|
| customers | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%); others 0 |
| products | 40 x 4 | none |
| orders | 13084 x 9 | order_amount 53 (0.41%, the literal text `N/A` that pandas reads as missing), delivery_minutes 2541 (19.42%); others 0 |

Dtypes: text columns are strings, `age` and `delivery_minutes` load as float (they contain blanks), `quantity`, `discount_pct`, `price_inr` as integers, and `order_amount` as text because it holds `error` entries.

| Audit figure | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount (53 `N/A` + 67 `error`) | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules and how much each changed

**Customers**
1. `city`: strip spaces and title-case, 24 spellings become 6. Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
2. `gender`: `male/m/M/Male/MALE` become male, `female/f/F/Female/FEMALE` become female, blank becomes unknown. Result: female 610, male 560, unknown 30.
3. `age`: the 6 values outside 10-100 become missing; all 67 missing ages (61 blank + 6) are filled with the median age of the customer's city (`groupby(...).transform("median")`). Result: 0 missing, all ages 18 to 60.
4. `signup_date` converted to datetime.

**Orders** (in this order)
1. `order_ts` converted to datetime.
2. Exact duplicates removed: 119 rows.
3. For each `order_id` keep the latest `order_ts`: 656 earlier pending records removed.
4. `order_amount` converted to numeric; the 112 unparseable amounts that survived steps 2 and 3 (120 in the raw file) are filled with `round(price_inr x quantity x (1 - discount_pct/100), 2)`.
5. Negative `delivery_minutes` set to missing (105 on surviving rows). For completed orders, 774 missing delivery times (blank plus the negatives) are filled with the median of the product category among completed orders. For cancelled orders the 1,199 missing values are left missing because a cancelled order was never delivered.

Result: 12,309 rows, 11,110 completed, 1,199 cancelled, no pending rows, 0 missing amounts, 0 missing delivery times among completed orders, completed revenue INR 2,998,668.80.

### Storage and validation

Clean tables are in `data/clean/*.csv` and in `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text). `src/part_a_validate.py` asserts 16 checkpoints (row counts, status counts, revenue, city and gender counts, missing values, database tables) and prints `ALL CHECKPOINTS PASSED`.

### SQL results (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; revenues sum to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average total of INR 2,552.06 |
| Q6 ranking windows | 8 rows have RANK() different from ROW_NUMBER() |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).

Ties make RANK() differ from ROW_NUMBER() because tied customers share one rank (and the next rank is skipped) while ROW_NUMBER() always gives every row a distinct number.

### pandas cross-check

`pd.pivot_table(..., margins=True)` gives a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory statistics (completed order_amount)

| Statistic | Value |
|---|---|
| Mean / median | 269.91 / 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 / IQR | 64.80 / 297.00 / 232.20 |
| Fences (1.5 x IQR) | lower -283.50, upper 645.30 |
| Outliers below / above | 0 / 897 |

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

**Answers (under 200 words).** The data is six months (April to September 2025) of synthetic quick-commerce orders: 11,110 completed orders from 1,175 customers in six cities. Order amounts are strongly right-skewed (mean 269.91 against median 135.00, skewness 4.61). By the 1.5 x IQR rule 897 orders lie above the upper fence of 645.30 and none below the lower fence; 635 of them are Electronics and the rest are mostly multi-unit baskets, so they are genuine large orders rather than errors. The real anomalies (duplicates, negative delivery times, unparseable amounts) were fixed during cleaning. Relationships are weak: the pair with the largest absolute correlation is order_amount with discount_pct (-0.095), just ahead of order_amount with quantity (+0.087), and delivery_minutes is essentially uncorrelated with everything. A correlation this small is negligible, and correlation of any size does not establish causation.

## Part B

### RFM (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19, min 0, max 180. Frequency: mean 9.46, max 33. Monetary: mean 2,552.06, max 14,766.55.

### K-means (StandardScaler, `KMeans(n_init=10, random_state=42)`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The elbow is at K = 4 (inertia falls 41% from K = 3 to K = 4 but only 17% from K = 4 to K = 5), K = 4 has the best silhouette (0.526), and four segments are a number the business can act on.

Inertia is the sum of squared distances to the nearest centroid, so adding clusters can only lower it (with one cluster per customer it would be 0); K = 8 therefore always beats K = 2, and a falling inertia alone cannot choose K. We need the point where the gain flattens, plus silhouette and interpretability.

### Segment profile (K = 4)

Cluster numbers are only labels with no ordering; names are assigned from the profile.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders (is_premium = True) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics launches, free-delivery perks |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points, subscription for staples |
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Bundle coupons to lift basket size and frequency |
| 3 | Lapsed / Dormant | 182 | 109.04 | 2.20 | 685.99 | Win-back discount before they are lost |

PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`; Premium is true only for the highest-mean-monetary segment.

### Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with a completed order before the cutoff. Label `lapsed` = no completed order from 2025-09-01 to 2025-09-30: 443 lapsed (38.32%). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train, 232 test.

Features (all from orders dated before the cutoff plus the customer table): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days.

**No leakage.** No order dated 2025-09-01 or later entered any feature; the segment labels from Part B task 3 were not used as features; every encoder and scaler is inside a scikit-learn `Pipeline` and was fitted on training rows only (inside `GridSearchCV` they are refitted on each training fold); the oversampler was applied to training rows only. No encoder, scaler or resampler saw test rows. Nominal columns use `OneHotEncoder(drop="first")` (k-1 columns). The code asserts that no feature column has a missing value (0 missing).

The scaler's learned mean of frequency is 8.115, against 8.029 for the full cohort: the scaler only saw the 924 training rows, so its mean differs slightly from the cohort mean.

### Baseline and tuned models

Baseline (always "not lapsed") on the test set: accuracy 61.6%, lapsed-class F1 0.00.

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100}, cv=5, scoring f1 | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth {3, 5, 8} x min_samples_leaf {5, 10, 20}, cv=5, scoring f1 (45 fits) | depth 8, leaf 10 | 0.681 |

Test set (lapsed = positive class; confusion matrix rows = true, columns = predicted, order [not lapsed, lapsed]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

The best logistic C is 100 rather than the reference's 1; the brief allows this because the feature set is allowed to differ and the F1 is of the same order.

**Selected model: random forest** (test F1 0.655, above the 0.55 requirement). It has the higher F1 and precision at the same recall. Missing a lapsing customer costs that customer's future orders, while wrongly targeting a loyal one costs only a small incentive, so recall matters more than precision here; both models catch the same 58 of 89 lapsers, so the forest wins because it wastes fewer offers (30 against 32 false alarms). If the team wants to catch even more lapsers, the balanced logistic model below is the option.

### Class-balance experiment

Training lapsed rows duplicated with replacement (`random_state=42`) until 570 against 570, then the best logistic regression refitted; test set unchanged.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20); precision fell slightly. For the retention team this means catching far more of the customers who are about to leave, at the price of contacting more customers who would have stayed anyway, which is acceptable when an offer is cheap.

### Interpretation

Top 5 random-forest features: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery 0.061, cancel_rate 0.052. Business reading: how often and how much a customer orders drives lapse most; the days since the last order matters next; slow deliveries and cancellations add a smaller signal.

Logistic-regression coefficients (standardised features): frequency -1.697 (more orders means much less likely to lapse), monetary +0.283 (for the same number of orders, high-value customers, who order big items rarely, lapse a little more; frequency and monetary are correlated, so read the sign with care), recency +0.039 (a longer gap since the last order raises lapse risk, weakly).

The selected model is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly (checked in code).

**Retention list.** `outputs/retention_list.csv` holds the 10 Premium test customers with the highest predicted lapse probability (columns `customer_id, lapse_probability, actually_lapsed`); 1 of the 10 truly lapsed. The Premium flag comes from whole-window RFM that includes September, so it is a post-hoc whole-window label: the retention list is illustrative and not a leakage-free evaluation.

## Part C

### Guardrails and redaction

4 empty tickets get `skipped_empty`, 4 long tickets (each estimated at 734 tokens, limit 400) get `flagged_long`; neither is sent to the model. Redaction (email, then phone, then order id, with the exact patterns and placeholders in the brief):

| Type | Substitutions |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches, and an assertion in the code checks that no email, phone or order id appears in any prompt sent to the model. The user message is exactly the redacted ticket text.

### Prompts, retries, validation

Zero-shot prompt: 29 estimated tokens, no `EXAMPLES:`. Few-shot prompt: 114 estimated tokens, with an `EXAMPLES:` line and four labelled examples including a fraud one. Both contain `CLASSIFY`, the four categories and the JSON-only format.

Retries: maximum 3 per ticket, wait `0.05 x 2**n x uniform(0.9, 1.15)` seconds, one log line per retry (`retry log: ticket=T0002 retry=0 wait=0.0490s`). Each full run over the 156 eligible tickets has exactly 19 transient errors, all recovered; every logged wait lay between 0.0455 and 0.0568 s (an assertion checks the range). No ticket ended with a permanent error.

Validation: valid JSON with an allowed category is `classified_json`; otherwise the lowercase words from `re.findall(r"[a-z]+", ...)` must contain exactly one allowed category for `classified_keyword`; everything else (including JSON with a category outside the list) is `flagged_invalid_output` and goes to a human. Each full run: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output (plus 4 skipped_empty, 4 flagged_long).

### Zero-shot versus few-shot (no token budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 8,932 | 60.8 |
| Few-shot | 147 | 100.0% (147/147) | 22,192 | 151.0 |

Zero-shot confusion (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 90 more tokens per classified ticket (151.0 against 60.8, roughly 2.5 times, 13,260 tokens more in total) but removes all 21 errors. It is worth paying for when a wrong label is costly: 11 fraud tickets went to billing in the zero-shot run, which means they would miss the security team's 2-hour target. For low-stakes labels where 85.7% is enough and volume is high, the zero-shot prompt is the cheaper choice.

### Budget demo and final routing

`TOKEN_BUDGET = 6000` with the zero-shot prompt (`outputs/triage_budget_demo.csv`): 51 tickets end `budget_exhausted` (6,023 tokens used before the budget stopped further calls; the check happens before each call, so the last call can overshoot slightly).

Final run (`outputs/triage_results.csv`, 164 rows): few-shot prompt, `TOKEN_BUDGET = 100000`, routing from `outputs/customer_segments.csv` (customers missing from it count as not premium).

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

**What was rule-validated and what went to humans.** 147 tickets passed the output rules (135 JSON, 12 keyword) and were routed automatically. 17 went to human review: 4 empty, 4 too long and 9 whose model output was invalid (never guessed). In the budget demo, the exhausted tickets also go to human review.

## Part D

### Chunking and retrieval

40-word chunks with 10-word overlap give 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. For the query `refund policy late delivery` the top three chunks all come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks so rare, informative words count more. A limitation is that TF-IDF matches words, not meaning: "money back" and "refund" are unrelated to it, so a question phrased with different words from the policy can miss the right chunk.

### Evaluation (14 questions: 12 answerable, 2 not covered)

| Setting | expected_doc in top 3 (of 12) | answer contains expected_fact (of 12) | fallback "I could not find" (of 2) |
|---|---|---|---|
| Overlap 10 (15 chunks) | 12 | 12 | 2 |
| Overlap 0 (13 chunks) | 12 | 10 | 2 |

With zero overlap two questions break because the sentence holding the answer is cut at a chunk boundary: "What delivery time makes a completed order eligible for a full refund?" returns only "receives a full refund of the order amount." (the phrase "more than 45 minutes" sits in the previous chunk), and "How soon is money returned when a payment fails but is deducted?" loses "48 hours" in the same way.

### Tools

Five tools in `TOOLS_SPEC` and `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (returns the retrieval results), `calculate_refund` (100% of the amount if delivery took more than 45 minutes, else 0), `issue_refund` and `escalate_to_human` (each appends a timestamped line to `outputs/refund_ledger.jsonl`). Refunds above INR 1000 are escalated, never issued.

### Seven scenarios (`outputs/agent_runs.jsonl`)

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,127 |
| S2 | REFUND | 135.00 | 5 | 1,127 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,114 |
| S4 | DENIED | 0.00 | 4 | 789 |
| S5 | DENIED | 0.00 | 3 | 475 |
| S6 | NOT_FOUND | 0.00 | 2 | 164 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,276 |

All refund amounts equal `expected_refund_inr` within 0.01. S7 uses all six model calls under the default `token_limit=6000`.

### Stop-safely demonstrations (`outputs/agent_demos.jsonl`)

| Demo | Status |
|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT (492 tokens after the third call; the tool it requested was not executed) |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT (stops before the first model call, 0 calls) |

Every fallback returns the steps completed so far and a short hand-over summary in `final_answer`.

### Crash and resume (S1)

A checkpoint in `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed, plus steps and call count) is written after every tool observation. A simulated crash was raised right after the second tool execution; `resume_agent("S1-crash-demo")` continued from the checkpoint. Tools executed across the crash and the resume, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated). Final status REFUND with INR 128.25.

### ReAct versus plan-and-execute

Measured ReAct cost: 5 model calls for S1 (1,127 tokens), 4 for the on-time order S4, 3 for the cancelled order S5 (475 tokens) and 2 for the missing order S6 (164 tokens). ReAct sends the whole history at every call, so cost grows with every step. A plan-and-execute design asks once for the step list (about 2 calls in total: plan plus final answer) and carries no growing history, so on the happy path it would use fewer calls and fewer tokens. Its weakness is unexpected observations: for NOT_FOUND or a cancelled order the pre-written plan is wrong after step 1, so it must detect the error and replan with an extra model call, whereas ReAct simply stops at call 2 or 3 at no extra cost. I would choose ReAct for this refund workflow: the path is short (at most five tool steps), each decision (completed or not, more than 45 minutes or not, above INR 1000 or not) depends on the latest observation, and money-moving actions should be decided after seeing the data, not planned ahead. The token saving of a plan is a few hundred tokens per ticket, while ReAct also gives a natural per-step checkpoint for resuming.

## Limitations and Next Steps

- All data and the language model are synthetic or mocked: the mock classifier is keyword-based and the mock agent follows a fixed script, so accuracies (85.7% and 100%) say little about real model quality.
- The Premium flag is a post-hoc whole-window label (its RFM includes September, the window that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation (only 1 of its 10 customers lapsed). A leakage-free version would compute Premium from orders before the cutoff.
- The lapse model uses one train/test split of 232 test rows and a 0.5 decision threshold; its test F1 (0.655) is noisy. Next steps: repeated cross-validation, threshold tuning for a chosen recall target, and calibrated probabilities.
- Delivery times for completed orders with missing values are imputed with the category median, which hides real variation; K-means segments depend on K = 4 and on standardised RFM only.
- Retrieval is TF-IDF, so it matches words rather than meaning; next steps are synonym handling or embeddings. The refund agent does not check the 7-day request window (treated as satisfied by the brief) and executes only one tool call per model reply.
- The virtual-environment install could not be tested in the build sandbox (no package index access), so `requirements.txt` pins the versions that were used to run everything (pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0).
