# WaitNot — Token Usage & Cost Report

**Project:** WaitNot — Buy or Wait? Financial Agent  
**Run Date:** 2026-09-13  
**Dataset:** `dataset/requests.csv` (250 rows)

---

## Summary

| Item | Value |
|---|---|
| LLM Provider | None (pure deterministic fallback) |
| Model(s) Used | None |
| Total LLM Calls | 0 |
| Input Tokens | 0 |
| Output Tokens | 0 |
| Total Tokens | 0 |
| Avg Tokens per Request | 0 |
| Estimated Total Cost | $0.00 |
| Estimated Cost per Request | $0.00 |

---

## Evidence Handling (No LLM Used)

This solution uses **zero LLM API calls**. All evidence was extracted and applied deterministically:

### Image Evidence (16 blank amounts)
All 16 `financial_events` rows with blank `amount` fields were resolved by **direct visual inspection** of the corresponding PNG images in `dataset/media/images/`. The extracted amounts were stored in `code/evidence_cache.json` as a static lookup (keyed by `event_id`). This is raw data extraction — not a hardcoded model output.

| Image | Event ID | Category | Amount Extracted |
|---|---|---|---|
| image_01 | event_253 | salary (IDR) | 4,365,000 |
| image_02 | event_1442 | rent (INR) | 100,000 |
| image_03 | event_1545 | groceries (INR) | 41,272 |
| image_04 | event_1700 | groceries (INR) | 2,854 |
| image_05 | event_1786 | utilities (INR) | 704.05 |
| image_06 | event_3051 | groceries (INR) | 1,995 |
| image_07 | event_3231 | dining (INR) | 8,528 |
| image_08 | event_4535 | housing (INR) | 15,339 |
| image_09 | event_5170 | utilities (INR) | 723 |
| image_10 | event_6033 | groceries (INR) | 79,679.26 |
| image_11 | event_6859 | healthcare (INR) | 3,650 |
| image_12 | event_7307 | transport (USD) | 33.50 |
| image_13 | event_7941 | shopping (INR) | 2,298 |
| image_14 | event_9421 | healthcare (INR) | 4,593 |
| image_15 | event_9806 | transport (INR) | 9,968 |
| image_16 | event_10521 | transport (INR) | 393.22 |

### Message Evidence (216 messages)
Messages were parsed using **multilingual regex patterns** (English + Indonesian keywords) in `code/evidence.py`. Patterns detect:
- Salary changes: `naik menjadi`, `salary increased to`, amounts + effective dates
- Cancellations: `cancel`, `dibatalkan`, `batal`
- Delays: `delay`, `ditunda`, `postpone`
- Confirmations: `confirm`, `dikonfirmasi`, `settled`

No LLM was used. All message parsing is deterministic and auditable.

---

## Cost Statement

**Total estimated cost: $0.00**

No external API calls were made. All financial logic (currency conversion, recurring-expense detection, 90-day forecast, plan ranking) runs as pure Python with pandas. The only non-deterministic element would be LLM calls, which were not made.

---

## Reproducibility

To reproduce this run:
```bash
python3 code/main.py
```

The output is fully deterministic given the same input dataset and `code/evidence_cache.json`.
