"""
evaluation/main.py — WaitNot Financial Agent
Score output.csv against the 25 known-good rows in dataset/sample_requests.csv.

Run: python3 code/evaluation/main.py
IMPORTANT: This file is the ONLY place that reads dataset/sample_requests.csv.
It is NEVER used as a training signal or lookup by code/main.py.
"""
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

SAMPLE_PATH = REPO_ROOT / "dataset" / "sample_requests.csv"
OUTPUT_PATH = REPO_ROOT / "output.csv"

NUMERIC_FIELDS  = ["amount_safe_to_pay"]
DATE_FIELDS     = ["earliest_date_for_full_payment"]
EXACT_FIELDS    = ["affordability_status", "recommended_payment_method",
                   "spending_changes_needed"]
PLAN_FIELD      = "payment_plan"


def parse_plan_amounts(plan_str: str) -> list[float]:
    """Extract sorted amounts from a payment_plan string."""
    if not plan_str or plan_str == "none":
        return []
    try:
        return sorted(float(leg.split(":")[1]) for leg in plan_str.split("|"))
    except Exception:
        return []


def score_amount(pred: float, gold: float, tol: float = 0.05) -> float:
    """1.0 if within 5% relative tolerance, else 0.0"""
    if gold == 0:
        return 1.0 if abs(pred) < 1e-6 else 0.0
    return 1.0 if abs(pred - gold) / abs(gold) <= tol else 0.0


def evaluate():
    if not SAMPLE_PATH.exists():
        print(f"[ERROR] sample_requests.csv not found at {SAMPLE_PATH}")
        sys.exit(1)

    import sys
    sys.path.insert(0, str(REPO_ROOT / "code"))
    from loaders import load_all, get_profile
    from events import get_user_events, classify_recurring, build_scheduled_events
    from evidence import parse_messages, apply_amendments
    from planner import choose_plan

    data = load_all()
    profiles_df = data["profiles"]
    events_df = data["events"]
    pay_opts_df = data["payment_options"]
    messages_df = data["messages"]
    fx_raw = data["fx_raw"]
    sample = data["sample_requests"]

    results = []
    for _, gold_row in sample.iterrows():
        rid = gold_row["request_id"]
        uid = gold_row["user_id"]
        try:
            profile = get_profile(profiles_df, uid)
            home_currency = profile["home_currency"]
            request = gold_row.to_dict()
            user_ev = get_user_events(events_df, uid, home_currency, fx_raw)
            recurring, one_time = classify_recurring(user_ev)
            scheduled = build_scheduled_events(user_ev)
            amendments = parse_messages(messages_df, uid, rid, home_currency)
            req_date = pd.Timestamp(gold_row["request_date"])
            recurring = apply_amendments(recurring, amendments, home_currency, fx_raw, req_date)
            
            pred_row = choose_plan(profile, request, pay_opts_df, recurring, scheduled, home_currency, fx_raw)
        except Exception as e:
            print(f"[WARN] Error on {rid}: {e}")
            continue

        gold_dict = gold_row.to_dict()
        row_result = {"request_id": rid}

        # Numeric fields
        for f in NUMERIC_FIELDS:
            try:
                p = float(pred_row.get(f, 0) or 0)
                g = float(gold_dict.get(f, 0) or 0)
                score = score_amount(p, g)
            except Exception:
                score = 0.0
            row_result[f + "_match"] = score

        # Exact string fields
        for f in EXACT_FIELDS:
            p = str(pred_row.get(f, "")).strip()
            g = str(gold_dict.get(f, "")).strip()
            row_result[f + "_match"] = 1.0 if p == g else 0.0

        # Date field
        for f in DATE_FIELDS:
            p = str(pred_row.get(f, "")).strip()
            g = str(gold_dict.get(f, "")).strip()
            row_result[f + "_match"] = 1.0 if p == g else 0.0

        # Payment plan (compare total amounts with tolerance)
        pp = parse_plan_amounts(str(pred_row.get(PLAN_FIELD, "")))
        pg = parse_plan_amounts(str(gold_dict.get(PLAN_FIELD, "")))
        if pp and pg and len(pp) == len(pg):
            plan_score = float(all(
                abs(a - b) / (abs(b) + 1e-9) <= 0.05
                for a, b in zip(pp, pg)
            ))
        elif pp == pg:  # both empty / both "none"
            plan_score = 1.0
        else:
            plan_score = 0.0
        row_result["payment_plan_match"] = plan_score

        results.append(row_result)

    if not results:
        print("No matching rows found in output.csv.")
        return

    df = pd.DataFrame(results)
    match_cols = [c for c in df.columns if c.endswith("_match")]

    print("=" * 60)
    print("  WaitNot -- Evaluation Summary (25 sample rows)")
    print("=" * 60)
    print(f"\n{'Field':<40} {'Accuracy':>10}")
    print("-" * 52)
    field_scores = {}
    for col in match_cols:
        acc = df[col].mean()
        field_scores[col] = acc
        field_name = col.replace("_match", "")
        print(f"{field_name:<40} {acc*100:>9.1f}%")

    overall = df[match_cols].mean(axis=1).mean()
    print("-" * 52)
    print(f"{'Overall (mean of all fields)':<40} {overall*100:>9.1f}%")
    print(f"\nRows evaluated: {len(df)} / {len(sample)}")

    # Per-row detail
    print("\n--- Per-row results ---")
    for _, r in df.iterrows():
        scores = {k.replace("_match",""): v for k, v in r.items() if k.endswith("_match")}
        row_avg = sum(scores.values()) / len(scores)
        failed = [k for k, v in scores.items() if v < 1.0]
        fail_str = ", ".join(failed) if failed else "ALL CORRECT"
        print(f"  {r['request_id']}: {row_avg*100:.0f}%  [{fail_str}]")


if __name__ == "__main__":
    evaluate()
