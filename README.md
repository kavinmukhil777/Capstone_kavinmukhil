# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected Parts, built on fully synthetic, fictional data (the brand name only frames the scenario; all money is in INR).

| Part | What it does | Main code | Main outputs |
|---|---|---|---|
| A | Audit and clean the raw exports, store them in SQLite, answer business questions in SQL and pandas | `src/part_a_clean.py`, `src/validate.py`, `src/part_a_analysis.py`, `sql/queries.sql` | `data/clean/*.csv`, `data/instamart.db` |
| B | RFM + K-means segments, PCA, lapse prediction | `src/part_b_segments.py`, `src/part_b_lapse.py` | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Ticket triage: guardrails, PII redaction, retries, output validation, token budget, routing by segment | `src/part_c_triage.py` | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Policy RAG assistant and a tool-calling refund agent with safety limits and crash-resume | `src/policy_rag.py`, `src/refund_agent.py` | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

`generate_data.py` and `mock_llm.py` are the supplied files and are committed unmodified. Every model call goes through `mock_llm.py`: no API key, account, network call or paid service is used anywhere. Libraries: standard library, pandas, numpy, scikit-learn, joblib. Each script prints its results; the committed copies of that output are in `outputs/reports/`.

Each Part consumes the previous one: Part B reads `data/clean/*.csv` (A), Part C reads `outputs/customer_segments.csv` (B), Part D's agent reads `data/instamart.db` (A).

## Reproduce

Run from the repository root (Python 3.11+; SQLite 3.25+ is bundled with Python).

```bash
# 1. create and activate an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
# 2. install pinned dependencies
pip install -r requirements.txt
# 3. generate the raw data (last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164)
python generate_data.py
mkdir -p outputs/reports state
# 4. Part A: audit + clean + store, validate, SQL and pandas analysis
python src/part_a_clean.py      | tee outputs/reports/part_a_clean.txt
python src/validate.py          | tee outputs/reports/validate.txt        # prints ALL CHECKPOINTS PASSED
python src/part_a_analysis.py   | tee outputs/reports/part_a_analysis.txt
# 5. Part B: segments, then lapse model
python src/part_b_segments.py   | tee outputs/reports/part_b_segments.txt
python src/part_b_lapse.py      | tee outputs/reports/part_b_lapse.txt
# 6. Part C: triage
python src/part_c_triage.py     | tee outputs/reports/part_c_triage.txt
# 7. Part D: RAG evaluation, then refund agent
python src/policy_rag.py        | tee outputs/reports/policy_rag.txt
python src/refund_agent.py      | tee outputs/reports/refund_agent.txt
```

Shortcut: after step 2, `python run_project.py` runs steps 3-7 in order.

Seeds: `random_state=42` for K-means, splits, forests and resampling; the triage retry jitter is seeded (`random.seed(7)`) so the retry log is repeatable. `requirements.txt` pins the versions used (pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0 and their dependencies). The whole chain was re-run from a freshly created virtual environment and every reference checkpoint below was matched.

## Part A

### Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Data types are as read by pandas with `keep_default_na=False`: text columns are strings, `price_inr`, `quantity` and `discount_pct` are integers. `order_amount` and `delivery_minutes` load as text because they contain the planted bad values (`N/A`, `error`, blanks). "Missing" means a blank cell.

| Table | Column | Missing | % |
|---|---|---|---|
| customers | age | 61 | 5.08 |
| customers | gender | 30 | 2.50 |
| customers | customer_id, city, signup_date | 0 | 0.00 |
| products | all four columns | 0 | 0.00 |
| orders | delivery_minutes | 2,541 | 19.42 |
| orders | order_id, customer_id, product_id, quantity, order_ts, status, discount_pct | 0 | 0.00 |
| orders | order_amount (blank) | 0 (but 120 are non-numeric text) | 0.00 |

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules and how many rows/values each changed

Customers
1. City: strip spaces, title case. 476 of 1,200 values changed; exactly six cities remain.
2. Gender: lower-case and map `m/male` to male, `f/female` to female, blank to unknown. 950 values changed.
3. Age: values outside 10-100 set to missing (6), then all missing ages (61 blank + 6 invalid = 67) filled with the median age of the customer's city via `groupby(...).transform("median")`.
4. `signup_date` converted to datetime.

Orders (in this order)
1. `order_ts` converted to datetime.
2. Exact duplicate rows removed: 119.
3. One record per `order_id`, keeping the latest `order_ts` (drops the earlier pending record): 656 removed.
4. `order_amount` to numeric, unparseable entries become missing (112 remain after the steps above; the other 8 of the 120 were on rows already removed), filled with `round(price_inr x quantity x (1 - discount_pct/100), 2)`.
5. Negative `delivery_minutes` set to missing (105 remain after de-duplication; 106 in the raw file). Missing delivery of completed orders (774 in total after this step) filled with the median of the order's product category among completed orders (`groupby` + `transform`). Cancelled orders are left missing (1,199) because a cancelled order was never delivered.

| Check after cleaning | Result | Reference |
|---|---|---|
| order rows | 12,309 (one per order_id) | 12,309 |
| completed / cancelled / pending | 11,110 / 1,199 / 0 | 11,110 / 1,199 / 0 |
| missing order_amount | 0 | 0 |
| missing delivery_minutes (completed / cancelled) | 0 / 1,199 | 0 / 1,199 |
| sum of completed order_amount | INR 2,998,668.80 | INR 2,998,668.80 |
| cities | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 | same |
| genders | female 610, male 560, unknown 30 | same |
| ages | 0 missing, all between 18 and 60 | same |

### Store and validate

Clean tables are written to `data/clean/{customers,products,orders}.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/validate.py` asserts 13 checkpoints (the ones in the table above) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result | Reference |
|---|---|---|
| Q1 WHERE then HAVING | 30 customers | 30 |
| Q2 three-table join | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 | same |
| Q3 scalar subquery | 11 products above the average price | 11 |
| Q4 LEFT JOIN | 25 customers without a completed order | 25 |
| Q5 two CTEs | 550 customers above the average spend of INR 2,552.06 | 550 / 2,552.06 |
| Q6 window ranking | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK() different from ROW_NUMBER() | same |
| Q7 running total | 183 daily rows, last cumulative revenue INR 2,998,668.80 | same |

Why ties make RANK() and ROW_NUMBER() differ: customers with the same total spend share one RANK() value (and the next rank is skipped), whereas ROW_NUMBER() must give every row a distinct number, so every tied row after the first gets a different value from its rank.

### pandas cross-check

`pd.pivot_table(..., margins=True)` of completed revenue by city x category has a grand total of INR 2,998,668.80, equal to the SQL total of Q2. Per city (`groupby` with named aggregation, completed orders):

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory statistics of completed order_amount

| Statistic | Value |
|---|---|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 / IQR | 64.80 / 297.00 / 232.20 |
| fences (1.5 x IQR) | lower -283.50, upper 645.30 |
| outliers below / above | 0 / 897 |

Correlation matrix (completed orders): order_amount-discount_pct -0.095, order_amount-quantity +0.087, order_amount-delivery_minutes -0.003, discount_pct-quantity 0.000, discount_pct-delivery_minutes -0.010, quantity-delivery_minutes -0.013.

**Answers (under 200 words).** *What is the data about?* 11,110 completed quick-commerce orders from April to September 2025 across six Indian cities, valued in INR. *Distributions?* Order value is strongly right-skewed: the mean (269.91) is about twice the median (135.00) and skewness is 4.61; half of all orders lie between 64.80 and 297.00. *Outliers?* 897 orders exceed the upper fence of 645.30 and none fall below the lower fence; about 71% of them are Electronics, so they are legitimately expensive items, not errors. The real anomalies (duplicates, pending records, negative times, text amounts) were removed in cleaning. *Relationships?* All correlations are tiny. The pair with the largest absolute correlation is order_amount and discount_pct at about -0.095, just ahead of order_amount and quantity at +0.087. A correlation this small is weak and does not establish causation.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55.

### K-means (StandardScaler, `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The elbow is at K=4: going from 3 to 4 clusters removes 525 units of inertia, from 4 to 5 only 126, and the curve is flat-ish afterwards. K=4 also has the highest silhouette (0.526) and gives four segments a marketing team can name and act on (below). Inertia at K=8 is always lower than at K=2 because it is the sum of squared distances to the nearest centroid, and every extra centroid can only shorten (never lengthen) those distances; at K equal to the number of customers it would reach zero. So a lower inertia alone just rewards over-splitting, which is why the elbow, the silhouette and interpretability decide K.

### Segment profile (K=4). Cluster numbers are only labels and carry no ordering.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium Big Spenders** (is_premium = true, highest mean monetary) | 138 | 24.12 | 6.77 | 7,129.70 | Personal outreach, early access to electronics launches, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and subscription-style bundles for daily staples |
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Basket-building offers, free-delivery threshold nudges |
| 3 | Dormant / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon, short survey on why they left |

PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium (true only for segment 1).

### Lapse problem

Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff = **1,156**; lapsed (no completed order 1-30 September) = **443 (38.32%)**. Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train / 232 test**. Baseline "always not lapsed" on the test set: accuracy **61.6%**, lapsed-class F1 **0.000**.

Features (all from orders dated before 2025-09-01 and the clean customer table): recency, frequency, monetary, average discount_pct of completed orders, cancellation rate (cancelled / all final-status orders), average delivery_minutes, favourite category, city, age, gender, tenure in days. 

**Leakage statement.** No order dated on or after 2025-09-01 entered any feature; the segment labels from the clustering were not used as features; the one-hot encoder, the scaler and the over-sampler never saw test rows (all preprocessing sits inside a scikit-learn `Pipeline` fitted on training rows only, and over-sampling is applied to the training rows only). The code asserts that no feature column has a missing value. Encoding uses `OneHotEncoder(drop="first")` (k-1 columns per variable).

The scaler's learned mean of `frequency` (training rows only) is 8.1147, while the full-cohort mean is 8.0285; they differ because the scaler has only seen the 924 training customers, and not the 232 test customers.

### Tuning and evaluation

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100}, cv=5, scoring f1 | C = 100 | 0.6506 |
| RandomForestClassifier(n_estimators=100, random_state=42) | max_depth {3,5,8} x min_samples_leaf {5,10,20}, cv=5, scoring f1 (45 fits) | max_depth 8, min_samples_leaf 10 | 0.6811 |

Test set (lapsed = positive class; confusion matrix rows are true [not lapsed, lapsed], columns predicted):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (always not lapsed) | [[143, 0], [89, 0]] | 0 | 0 | 0.000 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** |

**Selected model: random forest.** It has the better cross-validated and test F1 (0.655, above the 0.55 requirement) and catches the same number of lapsing customers as the logistic regression while wrongly flagging two fewer loyal ones. Missing a customer who is about to lapse costs the whole future margin of that customer, whereas wrongly targeting a loyal customer costs only a discount or a message; so recall matters more than precision here, and a forest that is at least as good on both is the safer choice. (The recall-oriented balancing experiment below shows how to push recall further if the business wants that.)

### Class-balance experiment (training rows only)

Training set over-sampled with replacement (`random_state=42`) to 570 / 570; best logistic regression refit and scored on the unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| un-balanced | 0.644 | 0.652 | 0.648 |
| balanced | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20); precision fell slightly rather than rising. For the retention team that means the balanced model finds many more of the customers who will lapse, at the price of contacting somewhat more customers who would have stayed.

### Interpretation, persistence and retention list

Top five random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery 0.061, cancel_rate 0.053 (customers who order often and recently are rarely lost; slow delivery and cancellations erode loyalty).

Logistic-regression coefficient signs (standardised features): frequency **negative** (-1.70; the more often someone orders, the less likely they lapse), recency **positive** (+0.04; the longer since the last order, the more likely they lapse, though the effect is small once frequency is in the model), monetary **positive** (+0.28; given the same frequency, big-ticket, infrequent buyers lapse more).

`outputs/lapse_model.joblib` holds the selected model (the whole fitted pipeline); reloading it with `joblib.load` reproduces the original test predictions exactly (checked: True).

`outputs/retention_list.csv`: of the 40 Premium customers in the test set, the 10 with the highest predicted lapse probability (0.487 down to 0.452) are listed; **1 of the 10 truly lapsed.** The Premium flag comes from the whole-window RFM of the segmentation, which includes September (the window that defines the lapse label), so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation.

## Part C

### Input guardrails and redaction

Tickets are processed in file order. 4 empty tickets get `skipped_empty`; 4 tickets whose `est_tokens(text)` > 400 (each estimates to 734) get `flagged_long`. Neither is ever sent to the model. Redaction applies, in order, `[\w.+-]+@[\w-]+\.[\w.]+` to `[EMAIL]`, `\b[6-9]\d{9}\b` to `[PHONE]`, `ORD-\d{6}` to `[ORDER_ID]`; the user message is exactly the redacted text. An assertion inside the call wrapper checks that none of the three patterns matches any message (system or user) sent to the model.

| Type | Substitutions |
|---|---|
| email | 55 |
| phone | 91 |
| order id | 132 |
| **total** | **278** |

Re-applying the three patterns to the redacted texts finds 0.

### Prompts

Zero-shot (38 estimated tokens) and few-shot (129 tokens) system prompts both contain the word CLASSIFY, list billing/delivery/technical/fraud and require a JSON-only reply `{"category": "...", "reason": "..."}`. The few-shot prompt has an `EXAMPLES:` line with four labelled examples including a fraud one; the zero-shot prompt does not contain `EXAMPLES:`.

### Retries and output validation

Every call goes through a retry wrapper: at most 3 retries per ticket, wait before retry n = `0.05 x 2**n x random.uniform(0.9, 1.15)`, one log line per retry (`outputs/retry_log.txt`, e.g. `retry ticket=T0002 retry_number=0 wait=0.0490s`). `mock_llm.reset()` is called before each full run. A full run over the 156 eligible tickets hit **19 TransientLLMErrors, all recovered** by a retry; with this mock every retry is retry number 0 and all waits lie in 0.0455-0.0568 s (inside 0.045-0.0575). No ticket ended with a permanent error.

Validation: valid JSON with an allowed category is `classified_json`; non-JSON text is searched with `re.findall(r"[a-z]+", text.lower())` and accepted as `classified_keyword` if exactly one allowed category word appears; everything else (including valid JSON with a category outside the list, such as "shipping") is `flagged_invalid_output` and goes to human review, never guessed.

| Status (one full run, 164 tickets) | Count |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot (no budget)

| Run | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| zero-shot | 147 | 85.7% (126/147) | 10,336 | 70.31 |
| few-shot | 147 | 100% (147/147) | 24,532 | 166.88 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | technical | fraud |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt costs about 97 extra tokens per classified ticket (166.88 vs 70.31, roughly 2.4 times the spend) but removes all 21 errors. It is worth paying when a wrong label is expensive: in the zero-shot run 11 of the 34 fraud tickets were sent to billing and would have missed the 2-hour security SLA, and 10 app-fault tickets were mis-filed. If tickets were low-stakes and tokens were tight, the cheaper zero-shot prompt would be acceptable. Note that both accuracies reflect the deterministic keyword behaviour of the mock, not a real model.

### Token budget, routing and output

**Budget demo:** zero-shot run with `TOKEN_BUDGET = 6000` (the check runs before each model call, so the run used 6,036 tokens in total): **65 tickets ended `budget_exhausted`** and went to human review (`outputs/triage_budget_demo.csv`).

**Final run:** few-shot, `TOKEN_BUDGET = 100000` (24,532 tokens used, nothing exhausted), routed with `outputs/customer_segments.csv` (a customer without a segment row counts as not premium) into `outputs/triage_results.csv` (164 rows; columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used).

| Route | Priority | Tickets |
|---|---|---|
| support_queue | NORMAL | 102 |
| security_team (fraud) | HIGH | 34 |
| premium_desk | HIGH | 11 |
| human_review | NORMAL | 17 |

**What was rule-validated and what went to humans.** 147 tickets (135 JSON + 12 keyword) were machine-classified and rule-validated against the allowed list. 17 went to human review: 9 invalid model outputs, 4 empty tickets and 4 over-long tickets (none of these 17 reached a final automated decision).

## Part D

### Chunking, retrieval and grounding

Documents are split into 40-word chunks with 10-word overlap (stop after the first chunk that reaches the last word): **15 chunks** (refund 4, delivery 3, escalation 3, payment 3, privacy 2), each stored with its source file. `TfidfVectorizer()` is fitted on the chunk texts and queries are ranked by cosine similarity (`retrieve(query, k=3)` returns source, score, text). For "refund policy late delivery" all top three chunks come from `refund_policy.txt`.

TF (term frequency) measures how often a word occurs in a chunk, IDF (inverse document frequency) down-weights words that appear in many chunks so rare, informative words count more. A limitation: TF-IDF matches words, not meaning, so a question phrased with different words from the policy (for example "money back" versus "refund") would score poorly.

`answer(question)` retrieves three chunks and sends `mock_llm.chat` a system message containing POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) and a user message `CONTEXT:` + the three chunks each prefixed with `[<file>]` on its own line + `QUESTION:` + the question; it returns the text and the sources.

| Evaluation (14 questions) | Overlap 10 | Overlap 0 |
|---|---|---|
| chunks | 15 | 13 |
| answerable: expected_doc in top 3 | 12 / 12 | 12 / 12 |
| answerable: answer contains expected_fact | 12 / 12 | 10 / 12 |
| unanswerable: fallback "I could not find..." | 2 / 2 | 2 / 2 |

With zero overlap two questions break: "What delivery time makes a completed order eligible for a full refund?" (the answer returned is "receives a full refund of the order amount.", because the sentence "If a completed order is delivered in more than 45 minutes, ..." is cut across a chunk boundary and the "more than 45 minutes" half sits in a different chunk) and "How soon is money returned when a payment fails but is deducted?" (the sentence is cut right before "within 48 hours"). The overlap keeps the whole sentence together in at least one chunk.

### Tools and the agent

Five tools with a `TOOLS_SPEC` list and a `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (the same retrieval as above), `calculate_refund` (100% of the amount if delivered in more than 45 minutes, otherwise 0), `issue_refund` and `escalate_to_human` (each append a timestamped line to `outputs/refund_ledger.jsonl`). `run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` is a ReAct loop: call the model, execute the tool it asks for, append the observation, repeat; the status is the word before the colon in the final answer. A refund above INR 1000 is escalated, never issued automatically. As a safety guard, `issue_refund` never pays the same order twice: a second call for an order already refunded is written to the ledger flagged `duplicate_suppressed` (this is visible in the last ledger line, caused by the crash-and-resume demo of S1 replaying an order already refunded in the batch run).

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,057 |
| S2 | REFUND | 135.00 | 5 | 1,057 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,044 |
| S4 | DENIED (delivered within 45 minutes) | 0.00 | 4 | 733 |
| S5 | DENIED (cancelled order) | 0.00 | 3 | 433 |
| S6 | NOT_FOUND | 0.00 | 2 | 136 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,192 |

Every refund amount equals `expected_refund_inr`. S7 uses 3,192 estimated tokens, so the default `token_limit=6000` does not stop it early. `outputs/agent_runs.jsonl` has one line per run (the seven scenarios plus the three demonstrations, marked by a `variant` field) with scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool, arguments and observation.

### Stopping safely

| Demonstration | Status |
|---|---|
| S1 with `token_limit=200` (stopped after the 3rd model call at 450 tokens, without executing the tool it asked for; 2 tools had run) | FALLBACK_TOKEN_LIMIT |
| S1 with `time_limit_seconds=0` (stops before the first model call) | FALLBACK_TIMEOUT |
| S7 with the defaults (6 model calls, no final answer) | FALLBACK_MAX_ITERATIONS |

Every fallback returns the steps done so far and a short human-handoff summary in `final_answer`.

### Crash and resume (S1)

After every tool observation the run writes `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed so far). A simulated crash is raised right after the second tool execution (checkpoint already saved); `resume_agent("S1-crash-resume")` loads the checkpoint and continues. Tools executed across the crash and the resume, in order, with none repeated:

1. get_order
2. search_policy  *(crash here)*
3. calculate_refund *(after resume)*
4. issue_refund *(after resume)*

Final status REFUND, INR 128.25, 5 model calls in total.

### ReAct versus plan-and-execute

My ReAct agent needs one model call per step: 5 for S1 (1,057 tokens), 4 for S4, 3 for S5 and 2 for S6, 6 for the looping S7. A plan-and-execute design asks the model once for the whole step list and once to summarise, so about 2 calls for S1 and fewer tokens, because the growing history (every observation is re-sent each call) is not re-sent five times. Its weakness is the unexpected observation: with a fixed plan, a NOT_FOUND order (S6) or a cancelled order (S5) means the remaining planned steps (policy search, refund calculation, issue) must be discarded or run wastefully until an error triggers a replan, and a plan made before seeing the amount cannot know whether to issue or escalate. ReAct decides each next step from the latest observation, so S6 stops after 2 calls and S5 after 3 with no wasted tools, and each step can be checked against the token, time and iteration limits and checkpointed. For this refund workflow I would choose ReAct: money moves, every action depends on what the database just returned (status, delivery time, the INR 1000 threshold), and the extra 2-3 calls per request cost only a few hundred estimated tokens.

## Limitations and Next Steps

- All data is synthetic, and `mock_llm.py` is keyword-based, so the triage accuracies (85.7% and 100%) and the policy answers describe the mock's rules, not a real language model; results with a hosted model would differ.
- The Premium flag is a post-hoc whole-window label: the RFM behind it includes September, the window that defines the lapse label, so `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation (only 1 of its 10 customers actually lapsed). A leakage-free version would compute the segments from orders before 2025-09-01 only.
- The lapse model is evaluated on one 232-row test split; with 89 lapsed test customers, differences of a few F1 points between models are within noise. Next steps: repeated cross-validation, probability calibration and a threshold chosen from the cost of a missed lapse.
- Tuning picked the logistic-regression C at the edge of the grid (C = 100); the grid could be extended.
- TF-IDF retrieval is lexical; embeddings or a synonym list would handle paraphrases. The 7-day refund window is assumed satisfied for all simulated requests and is not checked.
- Ledger and checkpoints are local files; a production agent would use a transactional store and idempotency keys for payments.
- Cluster ids and names are tied to the seed: if K or the seed changes, the names must be reassigned from the profile.
