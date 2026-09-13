"""
events.py — WaitNot Financial Agent
Classify financial events into recurring vs one-time.
Filter by status. Apply FX conversion. Build the per-user cash-flow picture.
"""
import pandas as pd
import numpy as np
from loaders import convert_amount

# Cash-flow status rules (SPEC.md §2.2)
CASH_STATUS = {"settled", "scheduled", "pending"}
IGNORE_STATUS = {"cancelled", "failed", "unrealized"}

# Recurring detection thresholds (SPEC.md §2.2, §6 risk #3)
MIN_OCCURRENCES      = 3          # need ≥3 settled occurrences
MIN_INTERVAL_DAYS    = 20         # minimum days between occurrences (relaxed from 25 to 20 for some loose-monthly)
MAX_INTERVAL_DAYS    = 40         # maximum days between occurrences
AMOUNT_STABLE_PCT    = 0.35       # amounts within ±35% of median (relaxed for noisy categories)

# Categories always treated as recurring regardless of frequency detection
ALWAYS_RECURRING_TYPES = {"subscription", "income"}  # event_type
ALWAYS_RECURRING_CATS  = {"salary", "rent", "utilities", "debt_repayment", "insurance",
                           "gym", "cloud_storage", "streaming", "music_subscription",
                           "delivery_membership"}


def get_user_events(events_df: pd.DataFrame, user_id: str,
                    home_currency: str, fx_raw: dict) -> pd.DataFrame:
    """
    Filter events for a user, convert amounts to home_currency.
    Drops non-cash (direction=non_cash) and ignored statuses.
    Returns DataFrame with extra column 'amount_home'.
    """
    ev = events_df[events_df["user_id"] == user_id].copy()
    # Drop cancelled/failed/unrealized
    ev = ev[ev["status"].isin(CASH_STATUS)]
    # Drop non_cash direction (investment valuations)
    ev = ev[ev["direction"] != "non_cash"]

    # Determine effective date: settlement_date if available else event_date
    ev["eff_date"] = ev["settlement_date"].combine_first(ev["event_date"])

    # FX conversion: use eff_date for rate lookup
    def _conv(row):
        cur = str(row["currency"]).strip()
        d = row["eff_date"] if pd.notna(row["eff_date"]) else pd.Timestamp("2020-01-01")
        try:
            return convert_amount(float(row["amount"]), cur, home_currency, d, fx_raw)
        except Exception:
            return float(row["amount"])  # fallback: no conversion

    ev["amount_home"] = ev.apply(_conv, axis=1)
    return ev


def classify_recurring(ev: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Split events into:
      - recurring: list of dicts with keys:
            event_type, category, amount, interval_days, last_date, flexibility, event_id, direction
      - one_time: list of event rows (as dicts)

    Recurring detection:
      1. event_type in ALWAYS_RECURRING_TYPES → always recurring
      2. category in ALWAYS_RECURRING_CATS → always recurring if ≥1 settled occurrence
      3. Otherwise: ≥MIN_OCCURRENCES settled debits with consistent interval & stable amount
    """
    settled = ev[ev["status"] == "settled"].copy()

    recurring_series = {}  # key → (interval_days, last_date, amount, flexibility, last_event_id)

    def _key(row):
        return (row["event_type"], row["category"], row["direction"])

    # Group by (event_type, category, direction)
    for key, group in settled.groupby([settled["event_type"], settled["category"], settled["direction"]]):
        event_type, category, direction = key
        group = group.sort_values("eff_date")

        # Always-recurring types
        if event_type in ALWAYS_RECURRING_TYPES:
            dates = group["eff_date"].dropna().tolist()
            amounts = group["amount_home"].tolist()
            if not dates:
                continue
            # Estimate interval
            if len(dates) >= 2:
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                interval = np.median(intervals)
                if interval < 1:
                    interval = 30
            else:
                interval = 30
            # Use median of last 3 amounts
            last3 = amounts[-3:] if len(amounts) >= 3 else amounts
            amount = float(np.median(last3))
            recurring_series[key] = {
                "event_type": event_type,
                "category": category,
                "direction": direction,
                "amount": amount,
                "interval_days": interval,
                "last_date": dates[-1],
                "flexibility": group["flexibility"].iloc[-1] if "flexibility" in group.columns else "fixed",
                "event_id": group["event_id"].iloc[-1],
            }
            continue

        # Always-recurring categories
        if category in ALWAYS_RECURRING_CATS and direction == "debit":
            dates = group["eff_date"].dropna().tolist()
            amounts = group["amount_home"].tolist()
            if not dates:
                continue
            if len(dates) >= 2:
                intervals = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]
                interval = float(np.median(intervals)) if intervals else 30.0
                interval = max(interval, 1.0)
            else:
                interval = 30.0
            last3 = amounts[-3:] if len(amounts) >= 3 else amounts
            amount = float(np.median(last3))
            recurring_series[key] = {
                "event_type": event_type,
                "category": category,
                "direction": direction,
                "amount": amount,
                "interval_days": interval,
                "last_date": dates[-1],
                "flexibility": group["flexibility"].iloc[-1] if "flexibility" in group.columns else "fixed",
                "event_id": group["event_id"].iloc[-1],
            }
            continue

        # General recurring detection
        if direction == "debit" and len(group) >= MIN_OCCURRENCES:
            dates = group["eff_date"].dropna().tolist()
            amounts = group["amount_home"].tolist()
            if len(dates) < MIN_OCCURRENCES:
                continue
            intervals = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]
            med_interval = float(np.median(intervals))
            if not (MIN_INTERVAL_DAYS <= med_interval <= MAX_INTERVAL_DAYS):
                continue
            # Check interval consistency (all within ±50% of median)
            if not all(abs(iv - med_interval) <= med_interval * 0.5 for iv in intervals):
                continue
            # Check amount stability
            med_amount = float(np.median(amounts))
            if med_amount == 0:
                continue
            if not all(abs(a - med_amount) <= med_amount * AMOUNT_STABLE_PCT for a in amounts):
                continue
            last3 = amounts[-3:]
            amount = float(np.median(last3))
            recurring_series[key] = {
                "event_type": event_type,
                "category": category,
                "direction": direction,
                "amount": amount,
                "interval_days": med_interval,
                "last_date": dates[-1],
                "flexibility": group["flexibility"].iloc[-1] if "flexibility" in group.columns else "fixed",
                "event_id": group["event_id"].iloc[-1],
            }

    recurring = list(recurring_series.values())

    # Build one_time: everything NOT captured by the recurring groups
    recurring_keys = set(recurring_series.keys())
    one_time = []
    for _, row in ev.iterrows():
        key = (row["event_type"], row["category"], row["direction"])
        if key not in recurring_keys:
            one_time.append(row.to_dict())

    return recurring, one_time


def build_scheduled_events(ev: pd.DataFrame) -> list[dict]:
    """
    Return all events with status='scheduled' or status='pending' as explicit
    one-time future debits/credits to apply on their settlement/event date.
    These are ground-truth future events — do not re-derive via recurrence.
    """
    sched = ev[ev["status"].isin({"scheduled", "pending"})].copy()
    result = []
    for _, row in sched.iterrows():
        result.append(row.to_dict())
    return result


def project_recurring(recurring: list[dict], start_date: pd.Timestamp,
                       end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, float, str, str, str]]:
    """
    Project recurring cash flows from start_date to end_date (inclusive).
    Returns list of (date, amount_home, direction, category, event_id).
    Amounts are positive; direction indicates debit/credit.
    """
    projected = []
    for rec in recurring:
        last_date = pd.Timestamp(rec["last_date"])
        interval = int(round(rec["interval_days"]))
        if interval < 1:
            interval = 30
        amount = rec["amount"]
        direction = rec["direction"]
        cat = rec["category"]
        eid = rec["event_id"]

        # Find first projected date after last_date within window
        next_date = last_date + pd.Timedelta(days=interval)
        while next_date <= end_date:
            if next_date >= start_date:
                projected.append((next_date, amount, direction, cat, eid))
            next_date += pd.Timedelta(days=interval)

    return projected
