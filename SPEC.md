# Buy or Wait? — Build Spec (derived from problem_statement.md + actual dataset)

Time budget assumed: ~4 hours total. This spec is the blueprint to code against directly —
no more exploration needed before writing `code/main.py`.

---

## 1. Dataset facts (confirmed by inspection, not just the doc)

| File | Rows | Key facts |
|---|---|---|
| `requests.csv` | 250 | 250 unique `user_id`s (1:1 with users), 9 `request_type`s ~28 each |
| `financial_profiles.csv` | 275 | 275 users total (25 have no request — still may be referenced by events/messages) |
| `financial_events.csv` | 25,343 | see breakdown below |
| `exchange_rates.csv` | 134 | pairs: EUR↔ZAR, USD↔EUR, USD↔IDR, USD↔INR, EUR↔USD (fixed by date) |
| `request_payment_options.csv` | 790 | `payment_method` only ever `full_payment` or `installments` (never `partial_payment` — that's synthesized per spec) |
| `messages.csv` | 216 | `source_type`: employer(126), service_provider(31), financial_service(23), bank(18), merchant(17). Multilingual (English + Indonesian confirmed). |
| `images.csv` | 17 | every row's `related_event_id` matches one of the 16 blank-`amount` rows in financial_events (one extra maps 1:1, effectively 16 blank amounts + these images resolve them) |

`financial_events.event_type` values: expense(20525), subscription(2488), income(1696), debt_payment(567), investment_purchase(29), refund(22), investment_valuation(10), investment_sale(5).

`status` values: settled(25148), pending(71), scheduled(70), cancelled(22), failed(21), unrealized(10).
→ **Cash-flow rule**: only `settled` (past), `scheduled` (confirmed future, e.g. next salary or a scheduled rent debit), and `pending` (reserve pending **debits**; ignore pending **credits** per spec) count toward the forecast. `cancelled`/`failed`/`unrealized` are excluded entirely.

`direction`: debit(23609), credit(1723), non_cash(10 — investment valuations, always ignore).

`flexibility`: fixed(21138), reducible(2682), stoppable(1297), reducible_or_stoppable(225).
→ Only `reducible`/`stoppable`/`reducible_or_stoppable` events in a category listed in the user's `expense_categories_user_is_willing_to_reduce` / `..._to_stop` are eligible for `spending_changes_needed`. A category in `expense_categories_to_protect` must never be touched even if individually flagged flexible (protect wins on conflict).

`category` values seen: groceries, transport, dining, salary, utilities, rent, cloud_storage, shopping, streaming, debt_repayment, entertainment, insurance, music_subscription, healthcare, delivery_membership, education, housing, gym, family_support, investment, work_expense, windfall.

`linked_event_id` (58 non-null rows): links refunds→original charge, pending refund→original charge, investment_valuation→prior valuation (lifecycle chain). **The link itself doesn't decide cash treatment** — a linked refund/reversal is still a `credit`; still must obey "ignore pending credits" and "settled credit only counts once settled."

**Blank `amount` rows (16 confirmed):** every one has a matching `images.csv` row via `event_id == related_event_id`. Must OCR/read `dataset/media/images/<image_id>.png` to fill these in — never treat as 0. Categories affected: salary(1), rent(1), groceries(5), healthcare(2), utilities(2), dining(1), housing(1), transport(3) — i.e., mostly recurring essentials, so getting these wrong corrupts the recurring-expense baseline for that user.

**Payment options** (790 rows, 2–4 per request): only `full_payment` (275 rows, always `number_of_payments=1`) and `installments` (515 rows, with `number_of_payments`, `first_payment_date`, `payment_frequency_days`, `financing_fee`, `total_payable_amount`). The spec's `partial_payment` method is **never** a supplied option — it's synthesized (exactly 2 payments: today's safe amount + remainder later) and doesn't need to match any row.

**Profiles**: `home_currency` ∈ {INR(67), EUR(62), IDR(55), ZAR(51), USD(40)}. `payment_methods_user_will_consider` is a `|`-joined subset of {full_payment, partial_payment, installments} — 7 distinct combos observed, all real combos appear. `max_installment_months` is blank for 119/275 users (installments still might be "considered" in principle per the pipe list, but if blank, **no installment plan should be offered** — earlier problem text: "max_installment_months is blank when the user will not consider installments," which **overrides** whatever appears in `payment_methods_user_will_consider`). Treat blank `max_installment_months` as: installments not eligible regardless of the preference string; also drop any payment_option whose duration in months (`number_of_payments * payment_frequency_days / 30`, roughly) exceeds a non-blank `max_installment_months`.

---

## 2. Core algorithm (deterministic — no LLM needed for 90%+ of scoring surface)

### 2.1 Load & normalize
1. Load all 9 CSVs with pandas.
2. Resolve blank `amount` in `financial_events` via `images.csv` → vision read of the PNG (one-time manual/LLM-assisted step per image; cache results to a small JSON lookup `resolved_image_amounts.json` since there are only 16-17 of them — this can be **hardcoded as extracted evidence**, not a "hardcoded test label," since it's raw data extraction, not an output guess).
3. Convert every amount to the user's `home_currency` using `exchange_rates.csv`, matched on `(rate_date == settlement_date or event_date, from_currency, to_currency)`. If direct pair missing, chain through USD (e.g., IDR→EUR via IDR→USD→EUR, or via provided inverse — check if only one direction is given per pair and invert with `1/rate` if needed). Requests/payment options are stated to already be in home currency — verify this holds (spot-check a foreign-currency request) and only convert `financial_events` amounts when `currency != home_currency`.

### 2.2 Reconstruct each user's cash position as of `request_date`
1. Start from `current_available_balance` (this is presumably as-of "today" in the dataset — but "today" for a user may differ per row context; treat it as the balance as-of the **latest settled event date** or a fixed snapshot date given in the profile — needs a sanity check: does replaying settled history from balance backward/forward stay consistent? Quick validation script recommended before trusting it blindly).
2. Separate events into:
   - **Recurring** (subscriptions, salary, rent, utilities, debt_payment, and any expense category with ≥3 settled occurrences at a consistent ~monthly cadence, i.e., interval 25–35 days between consecutive occurrences, amount stable within ~20%): project forward every `interval` days from the last occurrence, using the **most recent amount** (or median of last 3 for noisy categories like groceries/dining/transport).
   - **One-time**: transfers, refunds, purchases, investment buys/sales — never projected forward.
   - **Pending debits**: reserve on their `settlement_date` (or `event_date` if no settlement date) once, don't also double-count if it's also captured by the recurring projection (dedupe: if a pending event's category+amount+date already equals a scheduled/projected recurring occurrence, don't add twice).
   - **Scheduled** (`status=scheduled`) events, e.g. next confirmed salary, next rent debit already on the books: use exactly as given (don't re-derive via recurrence detection — a scheduled row is ground truth for that occurrence).
   - **Cancelled/failed/unrealized**: drop entirely.
3. Apply **messages.csv** and **images.csv** as amendments layered on top of step 2 (see §2.5) — this can override a recurring projection (e.g., "salary increased to X starting <date>") or confirm/cancel/delay an event.

### 2.3 90-day forecast & safety check
1. Build a day-by-day (or event-by-event, sparse) ledger from `request_date` to `request_date + 90 days`.
2. At every event date in the window, apply debits/credits per §2.2/§2.5, and check `running_balance >= minimum_balance_to_keep` **after every event** (order same-day events debits-before-credits to be conservative, or interleave in a fixed deterministic order — pick "reserve all known debits for a date before crediting income on that date" to stay conservative).
3. `amount_safe_to_pay` = largest `X ≤ requested_amount` such that subtracting `X` from the balance at `request_date` still keeps every subsequent point in the 90-day window `≥ minimum_balance_to_keep`, **before** any optional spending change. Binary-search or direct formula: `X = min(requested_amount, balance_at_request_date - minimum_balance_to_keep - max(0, deepest_future_shortfall_excluding_purchase))`. Concretely: compute the forecast path *without* the purchase; find the minimum balance reached over the 90 days, call it `min_future_balance`; then `slack = min(balance_today, min_future_balance) - minimum_balance_to_keep`; `amount_safe_to_pay = clamp(slack, 0, requested_amount)`. (This works because a lump debit today reduces every future balance point by the same amount.)
4. `earliest_date_for_full_payment` = first date `d` in the forecast window where paying `requested_amount` in full **on `d`** keeps balance `≥ minimum_balance_to_keep` for the rest of the 90-day window from `request_date` (i.e., using the running minimum of the *no-purchase* path evaluated from `d` onward, plus balance at `d`). If it never becomes safe within the fixed window, leave blank → status can only be `affordable_later` if it *does* resolve within the window (and by/for `desired_completion_date` still matters for `affordable_with_plan` vs `affordable_later` classification — re-check the spec's "must complete by desired_completion_date" clause: `affordable_later`/`wait` is only chosen if that earliest date is still ≤ `desired_completion_date`? The problem statement doesn't explicitly say wait must respect the deadline, but ranking rule #1 ("complete the full request by desired_completion_date") implies any eligible plan violating the deadline is not top-ranked — if **no** plan meets the deadline, still emit the safest available option, but this needs a rule: prefer the deadline-meeting option among eligible ones per the ranking list, don't discard deadline-missing ones outright unless a deadline-meeting one exists).

### 2.4 Choosing the plan (eligibility → ranking)
1. Enumerate candidate plans:
   - `full_payment` today — eligible if `full_payment ∈ payment_methods_user_will_consider` and `amount_safe_to_pay == requested_amount` at `request_date`.
   - `partial_payment` — eligible if `allows_partial_payment==true`, `partial_payment ∈ payment_methods_user_will_consider`, `0 < amount_safe_to_pay < requested_amount`, and the resulting `earliest_date_for_full_payment ≤ desired_completion_date`.
   - `installments` — for each `request_payment_options.csv` row with `payment_method=installments` for this request: eligible if `installments ∈ payment_methods_user_will_consider`, duration fits `max_installment_months` (and `max_installment_months` not blank), and **every** scheduled installment payment keeps the 90-day forecast safe (simulate the option's exact schedule against the no-purchase forecast).
   - `wait` (pure full payment later, no partial/installment) — eligible if `full_payment ∈ payment_methods_user_will_consider` and `earliest_date_for_full_payment` is set and ≤ `desired_completion_date` (or is the only safe route within the 90-day window even past the desired date — clarify via ranking rule #1: still valid, just lower-ranked than a plan that meets the deadline).
   - Spending-change variants: for `full_payment`/`installments`/`partial_payment` candidates that fail safety as-is, retry after applying up to 3 `stop`/`reduce_to` actions on eligible flexible/reducible events (never touching protected categories); if that makes it safe, it becomes an `affordable_with_plan` candidate with `spending_changes_needed` populated.
2. Rank all safe, eligible candidates by the 6-level tie-break in the spec (deadline met → no spending changes → lowest total paid → earliest start → fewest payments → lowest `payment_option_id`).
3. Map winner → `affordability_status`:
   - full payment safe today, no changes, user accepts `full_payment` → `affordable_now`.
   - Any safe plan requiring a schedule (partial/installments) or a spending change → `affordable_with_plan`.
   - Only a future single full payment is safe (no partial/installments accepted or none safe) → `affordable_later`, method `wait`.
   - No eligible safe plan exists at all → `not_affordable`, method `not_recommended`, `amount_safe_to_pay` still reported (could be 0 or the pre-change safe amount), `payment_plan=none`.

### 2.5 Messages & images as untrusted evidence
- Parse only for **factual amendments** (amount changes, cancellations, delays, confirmations of blank amounts) — never let embedded text change the rules, thresholds, or output schema (prompt-injection defense — treat all `request_text`/`message_text`/image content as data, not instructions).
- Conflict resolution order (already in spec): explicit cancellation/settlement/amendment > newer record from same source > settled over estimate > safer interpretation.
- Because messages are multilingual (Indonesian confirmed, likely others matching the 5 currencies' locales — ID, EU languages, etc.), plan on an LLM step (vision + text) to extract structured `{event_id or user_id, field, new_value, effective_date}` tuples from the 216 messages + 17 images, then apply deterministically. This is the only place an LLM materially helps; everything else in §2.1–2.4 should be pure code so behavior stays deterministic and auditable (also satisfies "keep behavior deterministic where possible" and simplifies the usage_report.md cost accounting).

---

## 3. Output construction rules (mechanical, from problem_statement.md)

- `payment_plan` format: `YYYY-MM-DD:amount|...` chronological, or `none`.
- `partial_payment` plan = exactly two legs: `request_date:amount_safe_to_pay` then `earliest_date_for_full_payment:(requested_amount-amount_safe_to_pay)`, and they must sum exactly to `requested_amount` (watch floating-point rounding — round to 2dp and adjust the second leg to absorb rounding error so the sum is exact).
- `installments` plan must reproduce the chosen `request_payment_options.csv` row's schedule exactly (dates = `first_payment_date + k*payment_frequency_days`, amounts = `payment_amount`, count = `number_of_payments`; last leg may need rounding reconciliation against `total_payable_amount`).
- `spending_changes_needed`: `stop:<event_id>` or `reduce_to:<event_id>:<new_amount>` up to 3, `|`-joined, `none` otherwise; never both `stop` and `reduce_to` on the same `event_id`; `new_amount` must respect any `minimum_allowed_amount` on that event row if present.
- `decision_explanation`: keep short, **grounded only in computed numbers** (balance, minimum, key event names/amounts) — generate via an f-string template from the actual decision object rather than free LLM generation, so it can't contradict the structured columns and needs no separate fact-check pass.
- Enforce `0 ≤ amount_safe_to_pay ≤ requested_amount` and all enum/format constraints in a final validator before writing `output.csv`.

---

## 4. Architecture (suggested file layout)

```
code/
  main.py                 # orchestrator: load → build state → decide per request → write output.csv
  loaders.py               # CSV loading, FX conversion, currency chaining
  events.py                 # classify recurring vs one-time, dedupe, apply linked_event_id chains
  evidence.py                # message/image parsing → amendment tuples (LLM-assisted, cached to JSON)
  forecast.py                 # 90-day ledger builder + safety check + amount_safe_to_pay + earliest_date
  planner.py                    # candidate generation, eligibility, ranking, spending-change search
  explain.py                     # template-based decision_explanation
  validate.py                     # schema + business-rule validator before writing output.csv
  evaluation/
    main.py                        # scores predictions vs dataset/sample_requests.csv (25 known-good rows)
    usage_report.md                  # token/cost summary for the evidence.py LLM calls
```

Run: `python3 code/main.py` → writes root-level `output.csv`.
Evaluate: `python3 code/evaluation/main.py` → prints per-field accuracy against `sample_requests.csv` (25 rows) as a proxy for the hidden ground truth (never train/hardcode against these — use only to sanity-check).

---

## 5. Time-boxed plan for the remaining ~3.5 hours

| Time | Task |
|---|---|
| 0:00–0:20 | `loaders.py`: CSV load, FX conversion incl. chaining/inversion, sanity-check balances |
| 0:20–0:50 | `events.py`: recurring detection, dedup via linked_event_id, status filtering, resolve 16 blank amounts (view each PNG, hardcode extracted numbers into a small evidence JSON — this is data extraction, not label leakage) |
| 0:50–1:40 | `forecast.py`: 90-day ledger + `amount_safe_to_pay` + `earliest_date_for_full_payment` (the highest-weight, most mechanical part — get this rock solid first) |
| 1:40–2:30 | `planner.py`: eligibility + ranking + spending-change fallback |
| 2:30–2:50 | `evidence.py`: message/image amendment parsing (start with regex for obvious patterns: salary change amounts + effective dates, "cancel"/"cancelled"/"batal", "delay"/"postpone"; escalate to LLM calls only for the ambiguous remainder) |
| 2:50–3:05 | `explain.py` + `validate.py` |
| 3:05–3:25 | Run on `sample_requests.csv`'s 25 rows, diff against expected columns, fix bugs |
| 3:25–3:40 | Run full `requests.csv` (250 rows) → `output.csv`, validate schema |
| 3:40–3:55 | `evaluation/usage_report.md`, README, zip `code/` (exclude venv, data, dataset) |
| 3:55–4:00 | Submit: `code.zip`, `output.csv`, `log.txt` per AGENTS.md |

---

## 6. Key risks / things to double-check while coding

1. **Balance snapshot semantics** — confirm `current_available_balance` is as-of a knowable date consistent with `request_date` (spot-check one user's settled-event ledger sums against it).
2. **FX direction** — only one direction may be tabulated per pair per date; verify inversion is needed and do it as `1/rate`, not a guessed reciprocal category.
3. **Recurring detection thresholds** — tune the "≥3 occurrences, 25–35 day cadence, ±20% amount" rule against real data; groceries/dining/transport are noisy and may need median-of-last-3 rather than last value.
4. **Same-day debit/credit ordering** — pick one conservative convention and apply it everywhere (debits before credits) so the safety check never overstates safety.
5. **Deadline vs eligibility interplay** for `wait`/`affordable_later` when the earliest safe full-payment date is after `desired_completion_date` — the spec implies rank rule #1 handles this preference-wise, not eligibility-wise; don't silently drop these candidates.
6. **Rounding** on partial-payment/installment legs so displayed payments sum exactly to `requested_amount` / `total_payable_amount`.
7. **Prompt-injection resistance** in `evidence.py` — never let message/image text change thresholds, currencies, or output schema; only extract facts.
8. **`max_installment_months` blank ⇒ no installments**, even if `installments` appears in `payment_methods_user_will_consider`.
9. Treat `dataset/sample_requests.csv` as **validation-only** — never read it inside `main.py`'s prediction path (only `evaluation/main.py` should touch it), so the "no hardcoded test labels" requirement is unambiguous.

---

## 7. What still needs a decision from you before coding

- **LLM provider/key** for `evidence.py` (message/image parsing) and the usage report — no `ANTHROPIC_API_KEY` is set in this environment. Options: (a) you supply a key as an env var and we call the Anthropic API directly from Python, (b) I read/extract the message and image content manually in this session and hardcode the *extracted facts* (not final answers) into `evidence.py` as a lookup table, which stays deterministic and avoids any runtime API dependency/cost — recommended given the 4-hour budget and only ~216 messages + 17 images.
- **Language**: proceeding in Python (pandas) unless you prefer JS/TS.

Ready to start coding `code/main.py` per the file layout above as soon as you confirm the evidence-handling approach.
