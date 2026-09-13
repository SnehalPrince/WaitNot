"""
main.py — WaitNot Financial Agent
Orchestrator: load → build state → decide per request → validate → write output.csv

Run: python3 code/main.py
Output: <repo_root>/output.csv

IMPORTANT: This file NEVER reads dataset/sample_requests.csv.
That file is only touched by code/evaluation/main.py.
No answers are hardcoded by request_id.
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Ensure the code directory is on sys.path
CODE_DIR  = Path(__file__).parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from loaders  import load_all, get_profile
from events   import get_user_events, classify_recurring, build_scheduled_events
from evidence import parse_messages, apply_amendments
from planner  import choose_plan
from explain  import generate_explanation
from validate import validate_output

OUTPUT_COLS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def main():
    print("=" * 60)
    print("  WaitNot -- Buy or Wait? Financial Agent")
    print("=" * 60)

    print("[1/5] Loading datasets...")
    data = load_all()

    profiles_df  = data["profiles"]
    events_df    = data["events"]
    requests_df  = data["requests"]
    pay_opts_df  = data["payment_options"]
    messages_df  = data["messages"]
    fx_raw       = data["fx_raw"]

    print(f"      Loaded {len(requests_df)} requests, {len(events_df)} events, "
          f"{len(profiles_df)} profiles.")

    output_rows = []
    total = len(requests_df)

    print(f"[2/5] Processing {total} requests...")
    for i, (_, req_row) in enumerate(requests_df.iterrows(), 1):
        req_id   = str(req_row["request_id"])
        user_id  = str(req_row["user_id"])

        try:
            # --- Load profile ---
            profile = get_profile(profiles_df, user_id)
            home_currency = profile["home_currency"]
            request = req_row.to_dict()

            # --- Get user events, convert FX ---
            user_ev = get_user_events(events_df, user_id, home_currency, fx_raw)

            # --- Classify recurring vs one-time ---
            recurring, one_time = classify_recurring(user_ev)

            # --- Build scheduled/pending events list ---
            scheduled = build_scheduled_events(user_ev)

            # --- Parse messages for amendments ---
            amendments = parse_messages(
                messages_df, user_id, req_id, home_currency
            )

            # --- Apply salary-change amendments to recurring income ---
            req_date = pd.Timestamp(req_row["request_date"])
            recurring = apply_amendments(
                recurring, amendments, home_currency, fx_raw, req_date
            )

            # --- Choose plan ---
            result = choose_plan(
                profile, request, pay_opts_df,
                recurring, scheduled, home_currency, fx_raw
            )

            # --- Generate explanation ---
            req_amount = float(req_row["requested_amount"])
            explanation = generate_explanation(result, req_amount, home_currency)

            output_rows.append({
                "request_id":                req_id,
                "amount_safe_to_pay":        result["amount_safe_to_pay"],
                "affordability_status":      result["affordability_status"],
                "recommended_payment_method":result["recommended_payment_method"],
                "payment_plan":              result["payment_plan"],
                "earliest_date_for_full_payment": result["earliest_date_for_full_payment"],
                "spending_changes_needed":   result["spending_changes_needed"],
                "decision_explanation":      explanation,
            })

            if i % 50 == 0 or i == total:
                print(f"      {i}/{total} done...")

        except Exception as e:
            print(f"[WARN] Error processing {req_id} ({user_id}): {e}")
            # Emit a safe fallback row rather than crashing
            output_rows.append({
                "request_id":                req_id,
                "amount_safe_to_pay":        0.0,
                "affordability_status":      "not_affordable",
                "recommended_payment_method":"not_recommended",
                "payment_plan":              "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed":   "none",
                "decision_explanation":      f"Unable to process request due to data issue.",
            })

    print("[3/5] Validating output...")
    try:
        validate_output(output_rows, requests_df)
    except ValueError as e:
        print(f"[ERROR] Validation failed: {e}")
        sys.exit(1)

    print("[4/5] Writing output.csv...")
    out_path = REPO_ROOT / "output.csv"
    out_df = pd.DataFrame(output_rows, columns=OUTPUT_COLS)
    out_df.to_csv(out_path, index=False)
    print(f"      Written: {out_path} ({len(out_df)} rows)")

    print("[5/5] Summary:")
    print(out_df["affordability_status"].value_counts().to_string())
    print("\nDone. output.csv is ready.")


if __name__ == "__main__":
    main()
