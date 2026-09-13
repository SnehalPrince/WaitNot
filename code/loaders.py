"""
loaders.py — WaitNot Financial Agent
Load all CSVs, resolve FX, fill blank event amounts from image cache.
"""
import os, json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATASET   = REPO_ROOT / "dataset"
CODE_DIR  = Path(__file__).parent


def load_all() -> dict:
    """
    Returns a dict with all dataframes and supplementary structures.
    Keys: profiles, events, requests, payment_options, exchange_rates,
          messages, images, sample_requests
    """
    data = {}

    data["profiles"]        = pd.read_csv(DATASET / "financial_profiles.csv", dtype=str)
    data["events"]          = pd.read_csv(DATASET / "financial_events.csv",   dtype=str)
    data["requests"]        = pd.read_csv(DATASET / "requests.csv",           dtype=str)
    data["payment_options"] = pd.read_csv(DATASET / "request_payment_options.csv", dtype=str)
    data["exchange_rates"]  = pd.read_csv(DATASET / "exchange_rates.csv",     dtype=str)
    data["messages"]        = pd.read_csv(DATASET / "messages.csv",           dtype=str)
    data["images"]          = pd.read_csv(DATASET / "images.csv",             dtype=str)
    data["sample_requests"] = pd.read_csv(DATASET / "sample_requests.csv",   dtype=str)

    # --- Resolve blank amounts in events from image cache ---
    with open(CODE_DIR / "evidence_cache.json", encoding="utf-8") as f:
        cache = json.load(f)["image_amounts"]

    ev = data["events"]
    mask_blank = ev["amount"].isna() | (ev["amount"].str.strip() == "")
    for eid, amount in cache.items():
        row_mask = ev["event_id"] == eid
        if row_mask.any():
            ev.loc[row_mask, "amount"] = str(amount)

    still_blank = (ev["amount"].isna() | (ev["amount"].str.strip() == "")) & (ev["event_type"] != "investment_valuation")
    if still_blank.any():
        remaining = ev.loc[still_blank, "event_id"].tolist()
        print(f"[WARN] Events still blank after cache: {remaining}. Setting to 0.")
        ev.loc[still_blank, "amount"] = "0"

    # --- Cast numeric columns ---
    ev["amount"] = pd.to_numeric(ev["amount"], errors="coerce").fillna(0.0)
    ev["event_date"]      = pd.to_datetime(ev["event_date"],      errors="coerce")
    ev["settlement_date"] = pd.to_datetime(ev["settlement_date"], errors="coerce")

    profiles = data["profiles"]
    profiles["current_available_balance"] = pd.to_numeric(profiles["current_available_balance"], errors="coerce")
    profiles["minimum_balance_to_keep"]   = pd.to_numeric(profiles["minimum_balance_to_keep"],   errors="coerce")
    profiles["max_installment_months"]    = pd.to_numeric(profiles["max_installment_months"],     errors="coerce")

    reqs = data["requests"]
    reqs["requested_amount"]    = pd.to_numeric(reqs["requested_amount"],    errors="coerce")
    reqs["request_date"]        = pd.to_datetime(reqs["request_date"],        errors="coerce")
    reqs["desired_completion_date"] = pd.to_datetime(reqs["desired_completion_date"], errors="coerce")
    reqs["allows_partial_payment"] = reqs["allows_partial_payment"].str.lower() == "true"

    po = data["payment_options"]
    po["payment_amount"]      = pd.to_numeric(po["payment_amount"],      errors="coerce")
    po["number_of_payments"]  = pd.to_numeric(po["number_of_payments"],  errors="coerce")
    po["first_payment_date"]  = pd.to_datetime(po["first_payment_date"], errors="coerce")
    po["payment_frequency_days"] = pd.to_numeric(po["payment_frequency_days"], errors="coerce")
    po["financing_fee"]       = pd.to_numeric(po["financing_fee"],       errors="coerce").fillna(0)
    po["total_payable_amount"]= pd.to_numeric(po["total_payable_amount"], errors="coerce")

    data["messages"]["sent_at"] = pd.to_datetime(data["messages"]["sent_at"], errors="coerce", utc=True)

    # --- Build FX lookup: (rate_date, from_cur, to_cur) -> rate ---
    rates_df = data["exchange_rates"].copy()
    rates_df["rate"]      = pd.to_numeric(rates_df["rate"], errors="coerce")
    rates_df["rate_date"] = pd.to_datetime(rates_df["rate_date"], errors="coerce")

    # Build a dict keyed (from_currency, to_currency) -> list of (date, rate) sorted by date
    from collections import defaultdict
    fx_raw: dict = defaultdict(list)
    for _, row in rates_df.iterrows():
        key = (row["from_currency"], row["to_currency"])
        fx_raw[key].append((row["rate_date"], row["rate"]))
    # Sort each list
    for k in fx_raw:
        fx_raw[k].sort(key=lambda x: x[0])

    data["fx_raw"] = dict(fx_raw)

    return data


def get_fx_rate(fx_raw: dict, from_cur: str, to_cur: str, date: pd.Timestamp) -> float:
    """
    Return the applicable exchange rate to convert from_cur → to_cur on `date`.
    Strategy:
      1. Look for direct pair, use the rate on or before `date` (most recent).
      2. Try inverse (1/rate).
      3. Chain through USD: from_cur→USD→to_cur.
    Returns 1.0 if same currency. Raises ValueError if no route found.
    """
    if from_cur == to_cur:
        return 1.0

    def _find_rate(fc, tc, d):
        """Find rate for (fc→tc) on or before date d. Returns None if not found."""
        pairs = fx_raw.get((fc, tc))
        if pairs:
            applicable = [(dt, r) for dt, r in pairs if dt <= d]
            if applicable:
                return applicable[-1][1]
        return None

    # Direct
    r = _find_rate(from_cur, to_cur, date)
    if r is not None:
        return r

    # Inverse
    r_inv = _find_rate(to_cur, from_cur, date)
    if r_inv is not None and r_inv != 0:
        return 1.0 / r_inv

    # Chain through USD
    if from_cur != "USD" and to_cur != "USD":
        r1 = get_fx_rate(fx_raw, from_cur, "USD", date)
        r2 = get_fx_rate(fx_raw, "USD", to_cur, date)
        return r1 * r2

    raise ValueError(f"No FX rate found: {from_cur} -> {to_cur} on {date.date()}")


def convert_amount(amount: float, from_cur: str, to_cur: str,
                   date: pd.Timestamp, fx_raw: dict) -> float:
    """Convert amount from from_cur to to_cur using the FX table."""
    if from_cur == to_cur:
        return amount
    rate = get_fx_rate(fx_raw, from_cur, to_cur, date)
    return amount * rate


def get_profile(profiles: pd.DataFrame, user_id: str) -> dict:
    """Return the financial profile for a user as a dict."""
    row = profiles[profiles["user_id"] == user_id]
    if row.empty:
        raise ValueError(f"No profile for {user_id}")
    r = row.iloc[0].to_dict()
    # Parse pipe-separated list fields
    for field in ["financial_priorities", "expense_categories_to_protect",
                  "expense_categories_user_is_willing_to_reduce",
                  "expense_categories_user_is_willing_to_stop",
                  "payment_methods_user_will_consider"]:
        val = r.get(field, "")
        if pd.isna(val) or str(val).strip().lower() == "nan":
            val = ""
        r[field] = [x.strip() for x in str(val).split("|") if x.strip()]
    return r
