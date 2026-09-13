"""
forecast.py — WaitNot Financial Agent
90-day balance forecasting, amount_safe_to_pay, earliest_date_for_full_payment.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from events import project_recurring

FORECAST_DAYS = 90


def build_ledger(
    start_balance: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    recurring: list[dict],
    scheduled_events: list[dict],
    home_currency: str,
) -> list[tuple[pd.Timestamp, float]]:
    """
    Build a sparse day-by-day (event-by-event) ledger.
    Returns sorted list of (date, running_balance) for every event date.
    Conservative ordering: on any given day, apply debits BEFORE credits.
    """
    # Collect all cash-flow events in the window
    cash_flows: list[tuple[pd.Timestamp, float]] = []  # (date, signed_delta)

    # 1. Projected recurring
    proj = project_recurring(recurring, start_date, end_date)
    for date, amount, direction, cat, eid in proj:
        delta = -amount if direction == "debit" else +amount
        cash_flows.append((date, delta))

    # 2. Scheduled / pending one-time events
    for ev_dict in scheduled_events:
        # Pending credits → IGNORE (per spec: reserve pending debits, ignore pending credits)
        if ev_dict.get("status") == "pending" and ev_dict.get("direction") == "credit":
            continue
        eff = ev_dict.get("eff_date")
        if pd.isna(eff):
            eff = ev_dict.get("event_date")
        if pd.isna(eff):
            continue
        eff = pd.Timestamp(eff)
        if eff < start_date or eff > end_date:
            continue
        amount = float(ev_dict.get("amount_home", 0) or 0)
        direction = str(ev_dict.get("direction", "debit"))
        delta = -amount if direction == "debit" else +amount
        cash_flows.append((eff, delta))

    # Sort: primary key = date; secondary key = sign (debits before credits = negative before positive)
    cash_flows.sort(key=lambda x: (x[0], x[1]))

    # Build running balance
    balance = start_balance
    ledger = [(start_date, balance)]
    for date, delta in cash_flows:
        balance += delta
        ledger.append((date, balance))

    return ledger


def compute_min_future_balance(ledger: list[tuple[pd.Timestamp, float]]) -> float:
    """Return the minimum balance reached across all ledger entries."""
    if not ledger:
        return 0.0
    return min(b for _, b in ledger)


def compute_amount_safe_to_pay(
    start_balance: float,
    requested_amount: float,
    min_balance_to_keep: float,
    ledger_no_purchase: list[tuple[pd.Timestamp, float]],
) -> float:
    """
    SPEC.md §2.3:
    slack = min(balance_today, min_future_balance) - minimum_balance_to_keep
    amount_safe_to_pay = clamp(slack, 0, requested_amount)
    """
    min_future = compute_min_future_balance(ledger_no_purchase)
    slack = min(start_balance, min_future) - min_balance_to_keep
    return float(np.clip(slack, 0.0, requested_amount))


def find_earliest_full_payment_date(
    start_balance: float,
    requested_amount: float,
    min_balance_to_keep: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    recurring: list[dict],
    scheduled_events: list[dict],
    home_currency: str,
) -> pd.Timestamp | None:
    """
    Find the first date d in [start_date, end_date] where paying requested_amount
    on d keeps balance >= min_balance_to_keep for the rest of the 90-day window.

    Strategy: scan day by day. For each candidate date d:
      - Recompute ledger from start_date to d (without purchase) → balance_at_d
      - slack_at_d = balance_at_d - min_balance_to_keep
      - Also check the minimum balance from d to end_date in the no-purchase ledger
      - If (slack_at_d >= requested_amount) AND (min_from_d_to_end - requested_amount >= min_balance_to_keep):
          → d is safe

    To avoid O(90^2) complexity, build the no-purchase ledger once and scan it.
    """
    # Build no-purchase ledger for the full window
    full_ledger = build_ledger(
        start_balance, start_date, end_date,
        recurring, scheduled_events, home_currency
    )

    # Build cumulative min from-right: min_from_right[i] = min balance from ledger[i] onward
    dates = [d for d, _ in full_ledger]
    balances = [b for _, b in full_ledger]
    n = len(balances)

    # Scan each day in [start_date, end_date]
    current = start_date
    while current <= end_date:
        # Find balance at current date from ledger
        # The ledger is sorted; find the last entry <= current
        bal_at_current = start_balance
        for i in range(n):
            if dates[i] <= current:
                bal_at_current = balances[i]
            else:
                break

        # Check if paying requested_amount at current keeps future safe
        if bal_at_current - requested_amount >= min_balance_to_keep:
            # Verify the post-purchase balance stays above minimum for rest of window
            # The purchase reduces balance by requested_amount on `current`.
            # All future events from `current` onward are unchanged.
            # min_balance_after_purchase = (min from current to end_date in ledger) - requested_amount
            min_from_here = bal_at_current
            for i in range(n):
                if dates[i] >= current:
                    min_from_here = min(min_from_here, balances[i])
            if min_from_here - requested_amount >= min_balance_to_keep:
                return current

        current += timedelta(days=1)

    return None


def check_installment_safety(
    start_balance: float,
    min_balance_to_keep: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    recurring: list[dict],
    scheduled_events: list[dict],
    home_currency: str,
    installment_schedule: list[tuple[pd.Timestamp, float]],
) -> bool:
    """
    Check if a given installment schedule is safe:
    every installment payment, combined with the no-purchase forecast, keeps
    balance >= min_balance_to_keep throughout the 90-day window.
    """
    # Build base ledger
    base_ledger = build_ledger(
        start_balance, start_date, end_date,
        recurring, scheduled_events, home_currency
    )
    # Inject installment cash flows (debits)
    extra = [(d, -amt) for d, amt in installment_schedule if start_date <= d <= end_date]
    all_flows = [(d, b - base_ledger[0][1]) for d, b in base_ledger]  # relative deltas

    # Replay from scratch including installment payments
    # Build combined cash flows
    base_flows: list[tuple[pd.Timestamp, float]] = []
    prev_b = start_balance
    for d, b in base_ledger[1:]:
        delta = b - prev_b
        base_flows.append((d, delta))
        prev_b = b

    combined = base_flows + extra
    combined.sort(key=lambda x: (x[0], x[1]))  # debits before credits

    balance = start_balance
    for d, delta in combined:
        if d > end_date:
            break
        balance += delta
        if balance < min_balance_to_keep:
            return False
    return True
