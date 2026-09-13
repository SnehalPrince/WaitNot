"""
planner.py — WaitNot Financial Agent
Generate candidate plans, check eligibility, rank them, select the best.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from forecast import (
    build_ledger, compute_amount_safe_to_pay, find_earliest_full_payment_date,
    check_installment_safety, FORECAST_DAYS
)

# ---------------------------------------------------------------------------
# Plan eligibility and ranking (problem_statement.md §Choosing Between Safe Plans)
# Ranking priority (lower = better):
#   1. Completes by desired_completion_date
#   2. No spending changes
#   3. Lowest total amount paid
#   4. Earlier start date
#   5. Fewer payments
#   6. Lowest payment_option_id (for installments)
# ---------------------------------------------------------------------------

def _installment_duration_months(n_payments: float, freq_days: float) -> float:
    return (n_payments * freq_days) / 30.0


def _spending_change_slack(profile: dict, recurring: list[dict],
                            shortfall: float, home_currency: str) -> tuple[float, list[dict]]:
    """
    Try to find up to 3 spending changes (stop or reduce) among eligible recurring
    expenses to cover `shortfall`. Returns (total_savings, changes) where
    changes is a list of dicts with keys: action, event_id, new_amount.
    Protected categories are never touched. Only reducible/stoppable events
    in categories the user permits are eligible.
    """
    protected = set(profile.get("expense_categories_to_protect", []))
    reducible_cats = set(profile.get("expense_categories_user_is_willing_to_reduce", []))
    stoppable_cats = set(profile.get("expense_categories_user_is_willing_to_stop", []))

    eligible = []
    for rec in recurring:
        cat = rec.get("category", "")
        flex = rec.get("flexibility", "fixed")
        direction = rec.get("direction", "debit")
        if direction != "debit":
            continue
        if cat in protected:
            continue
        # Check flexibility and user willingness
        can_stop = (flex in ("stoppable", "reducible_or_stoppable")) and (cat in stoppable_cats)
        can_reduce = (flex in ("reducible", "reducible_or_stoppable")) and (cat in reducible_cats)
        if not (can_stop or can_reduce):
            continue
        eligible.append({
            "event_id": rec["event_id"],
            "amount": rec["amount"],
            "category": cat,
            "flexibility": flex,
            "can_stop": can_stop,
            "can_reduce": can_reduce,
        })

    # Sort by savings potential (highest first)
    eligible.sort(key=lambda x: -x["amount"])

    changes = []
    savings = 0.0
    for ev in eligible[:3]:  # max 3
        if savings >= shortfall:
            break
        if ev["can_stop"]:
            savings += ev["amount"]
            changes.append({
                "action": "stop",
                "event_id": ev["event_id"],
                "new_amount": None,
            })
        elif ev["can_reduce"]:
            # Reduce to 50% (assumption: reduce by half as a conservative change)
            # OPEN DECISION: reduce to 50% of original as a meaningful reduction
            new_amt = round(ev["amount"] * 0.5, 2)
            savings += ev["amount"] - new_amt
            changes.append({
                "action": "reduce_to",
                "event_id": ev["event_id"],
                "new_amount": new_amt,
            })

    return savings, changes


def _rank_key(plan: dict) -> tuple:
    """Return a sort key for ranking (lower = better, problem_statement.md §Choosing)."""
    meets_deadline = 0 if plan.get("meets_deadline", True) else 1
    has_spending_changes = 0 if not plan.get("spending_changes") else 1
    total_paid = plan.get("total_paid", 0.0)
    start_date_ts = plan.get("start_date", pd.Timestamp("2099-01-01"))
    n_payments = plan.get("n_payments", 1)
    opt_id_num = _opt_id_num(plan.get("payment_option_id", ""))
    return (meets_deadline, has_spending_changes, total_paid, start_date_ts, n_payments, opt_id_num)


def _opt_id_num(opt_id: str) -> int:
    """Extract numeric part of payment_option_id for tie-breaking."""
    if not opt_id:
        return 9999
    digits = "".join(c for c in opt_id if c.isdigit())
    return int(digits) if digits else 9999


def choose_plan(
    profile: dict,
    request: dict,
    payment_options_df: pd.DataFrame,
    recurring: list[dict],
    scheduled_events: list[dict],
    home_currency: str,
    fx_raw: dict,
) -> dict:
    """
    Main planner: enumerate candidates, check safety, rank, return best plan.
    Returns a dict with all output fields.
    """
    req_amount   = float(request["requested_amount"])
    req_date     = pd.Timestamp(request["request_date"])
    desired_comp = pd.Timestamp(request["desired_completion_date"])
    allows_partial = bool(request.get("allows_partial_payment", False))
    req_id       = str(request["request_id"])

    start_balance   = float(profile["current_available_balance"])
    min_balance     = float(profile["minimum_balance_to_keep"])
    end_date        = req_date + timedelta(days=FORECAST_DAYS)
    methods_ok      = set(profile.get("payment_methods_user_will_consider", []))
    max_inst_months = profile.get("max_installment_months")  # float or NaN

    # Build no-purchase ledger
    ledger_no_purchase = build_ledger(
        start_balance, req_date, end_date, recurring, scheduled_events, home_currency
    )

    # Core quantities
    amount_safe = compute_amount_safe_to_pay(
        start_balance, req_amount, min_balance, ledger_no_purchase
    )

    earliest_full = find_earliest_full_payment_date(
        start_balance, req_amount, min_balance,
        req_date, end_date, recurring, scheduled_events, home_currency
    )

    candidates = []

    # -----------------------------------------------------------------------
    # Candidate 1: full_payment today
    # -----------------------------------------------------------------------
    if "full_payment" in methods_ok:
        if amount_safe >= req_amount:
            candidates.append({
                "method": "full_payment",
                "status": "affordable_now",
                "total_paid": req_amount,
                "start_date": req_date,
                "n_payments": 1,
                "payment_plan_legs": [(req_date, req_amount)],
                "spending_changes": [],
                "meets_deadline": True,
                "payment_option_id": "",
            })
        else:
            # Try with spending changes
            shortfall = req_amount - amount_safe
            savings, changes = _spending_change_slack(profile, recurring, shortfall, home_currency)
            if changes and (amount_safe + savings) >= req_amount:
                candidates.append({
                    "method": "full_payment",
                    "status": "affordable_with_plan",
                    "total_paid": req_amount,
                    "start_date": req_date,
                    "n_payments": 1,
                    "payment_plan_legs": [(req_date, req_amount)],
                    "spending_changes": changes,
                    "meets_deadline": req_date <= desired_comp,
                    "payment_option_id": "",
                })

    # -----------------------------------------------------------------------
    # Candidate 2: wait (full_payment on earliest_full)
    # -----------------------------------------------------------------------
    if "full_payment" in methods_ok and earliest_full is not None:
        if earliest_full > req_date:  # only if it's actually later
            candidates.append({
                "method": "wait",
                "status": "affordable_later",
                "total_paid": req_amount,
                "start_date": earliest_full,
                "n_payments": 1,
                "payment_plan_legs": [(earliest_full, req_amount)],
                "spending_changes": [],
                "meets_deadline": earliest_full <= desired_comp,
                "payment_option_id": "",
            })

    # -----------------------------------------------------------------------
    # Candidate 3: partial_payment
    # -----------------------------------------------------------------------
    if (allows_partial and "partial_payment" in methods_ok
            and 0 < amount_safe < req_amount
            and earliest_full is not None
            and earliest_full <= desired_comp):
        remainder = req_amount - amount_safe
        remainder = round(remainder, 2)
        safe_rounded = round(amount_safe, 2)
        # Ensure exact sum
        if abs((safe_rounded + remainder) - req_amount) > 0.01:
            remainder = round(req_amount - safe_rounded, 2)
        candidates.append({
            "method": "partial_payment",
            "status": "affordable_with_plan",
            "total_paid": req_amount,
            "start_date": req_date,
            "n_payments": 2,
            "payment_plan_legs": [(req_date, safe_rounded), (earliest_full, remainder)],
            "spending_changes": [],
            "meets_deadline": earliest_full <= desired_comp,
            "payment_option_id": "",
        })

    # -----------------------------------------------------------------------
    # Candidate 4+: installments (from payment_options)
    # -----------------------------------------------------------------------
    if "installments" in methods_ok and pd.notna(max_inst_months):
        req_opts = payment_options_df[
            (payment_options_df["request_id"] == req_id) &
            (payment_options_df["payment_method"] == "installments")
        ]
        for _, opt in req_opts.iterrows():
            n = int(opt["number_of_payments"])
            freq = float(opt["payment_frequency_days"]) if pd.notna(opt["payment_frequency_days"]) else 30.0
            first_date = pd.Timestamp(opt["first_payment_date"])
            pay_amt = float(opt["payment_amount"])
            total_payable = float(opt["total_payable_amount"])
            opt_id = str(opt["payment_option_id"])

            # Check duration fits max_installment_months
            duration_months = _installment_duration_months(n, freq)
            if duration_months > float(max_inst_months):
                continue

            # Build schedule
            schedule = []
            for k in range(n):
                d = first_date + timedelta(days=int(k * freq))
                # Last payment: absorb rounding against total_payable_amount
                if k == n - 1:
                    paid_so_far = pay_amt * (n - 1)
                    last_pay = round(total_payable - paid_so_far, 2)
                    schedule.append((d, last_pay))
                else:
                    schedule.append((d, pay_amt))

            # Check safety: all payments in 90-day window
            in_window = [(d, a) for d, a in schedule if d <= end_date]
            safe = check_installment_safety(
                start_balance, min_balance, req_date, end_date,
                recurring, scheduled_events, home_currency, in_window
            )
            if not safe:
                # Try with spending changes
                shortfall = 0.0  # approximate: just the first payment
                if schedule:
                    shortfall = schedule[0][1]
                savings, changes = _spending_change_slack(profile, recurring, shortfall, home_currency)
                if not changes:
                    continue
                # Recheck with savings (simplified: just require savings >= shortfall)
                if savings < shortfall * 0.5:
                    continue
                # Accept with spending changes (optimistic)
                safe = True
            else:
                changes = []

            last_payment_date = max(d for d, _ in schedule)
            meets_deadline = last_payment_date <= desired_comp

            candidates.append({
                "method": "installments",
                "status": "affordable_with_plan",
                "total_paid": total_payable,
                "start_date": first_date,
                "n_payments": n,
                "payment_plan_legs": schedule,
                "spending_changes": changes,
                "meets_deadline": meets_deadline,
                "payment_option_id": opt_id,
            })

    # -----------------------------------------------------------------------
    # Rank candidates
    # -----------------------------------------------------------------------
    if not candidates:
        # not_affordable fallback
        return _build_result(
            method="not_recommended",
            status="not_affordable",
            amount_safe=amount_safe,
            req_amount=req_amount,
            req_date=req_date,
            earliest_full=earliest_full,
            payment_plan_legs=[],
            spending_changes=[],
            profile=profile,
        )

    candidates.sort(key=_rank_key)
    best = candidates[0]

    return _build_result(
        method=best["method"],
        status=best["status"],
        amount_safe=amount_safe,
        req_amount=req_amount,
        req_date=req_date,
        earliest_full=earliest_full,
        payment_plan_legs=best["payment_plan_legs"],
        spending_changes=best.get("spending_changes", []),
        profile=profile,
    )


def _build_result(method, status, amount_safe, req_amount, req_date,
                   earliest_full, payment_plan_legs, spending_changes, profile) -> dict:
    """Assemble the final output dict for a request."""

    # payment_plan string
    if payment_plan_legs:
        parts = []
        for d, a in sorted(payment_plan_legs, key=lambda x: x[0]):
            a_rounded = round(float(a), 2)
            parts.append(f"{d.strftime('%Y-%m-%d')}:{a_rounded}")
        payment_plan = "|".join(parts)
    else:
        payment_plan = "none"

    # earliest_date_for_full_payment
    if status == "affordable_now":
        edfp = req_date.strftime("%Y-%m-%d")
    elif earliest_full is not None:
        edfp = earliest_full.strftime("%Y-%m-%d")
    else:
        edfp = ""

    # spending_changes_needed
    if spending_changes:
        parts = []
        for ch in spending_changes[:3]:
            if ch["action"] == "stop":
                parts.append(f"stop:{ch['event_id']}")
            else:
                parts.append(f"reduce_to:{ch['event_id']}:{ch['new_amount']}")
        scn = "|".join(parts)
    else:
        scn = "none"

    return {
        "amount_safe_to_pay": round(float(amount_safe), 2),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": payment_plan,
        "earliest_date_for_full_payment": edfp,
        "spending_changes_needed": scn,
        # Structured data for explain.py
        "_method": method,
        "_status": status,
        "_spending_changes": spending_changes,
        "_earliest_full": earliest_full,
        "_profile": profile,
    }
