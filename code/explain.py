"""
explain.py — WaitNot Financial Agent
Generate concise, grounded decision_explanation from structured plan data.
Uses f-string templates — no free-text LLM generation — so it cannot
contradict the structured columns.
"""
import pandas as pd


def generate_explanation(result: dict, req_amount: float, home_currency: str) -> str:
    """
    Generate a short decision explanation grounded in the computed numbers.
    The explanation is templated — not free-form — to ensure it matches the
    structured output columns exactly.
    """
    method  = result.get("_method", result.get("recommended_payment_method", ""))
    status  = result.get("_status", result.get("affordability_status", ""))
    safe    = result.get("amount_safe_to_pay", 0.0)
    edfp    = result.get("earliest_date_for_full_payment", "")
    scn     = result.get("spending_changes_needed", "none")
    profile = result.get("_profile", {})
    min_bal = float(profile.get("minimum_balance_to_keep", 0)) if profile else 0.0

    cur = home_currency
    fmt = lambda x: f"{cur} {x:,.2f}".rstrip("0").rstrip(".")

    if status == "affordable_now":
        return (
            f"Pay {fmt(req_amount)} today. "
            f"This leaves at least {fmt(min_bal)} available over the next 90 days."
        )

    elif status == "affordable_with_plan":
        if method == "installments":
            plan = result.get("payment_plan", "none")
            legs = [p for p in plan.split("|") if p != "none"] if plan != "none" else []
            n = len(legs)
            first_leg = legs[0] if legs else ""
            first_date = first_leg.split(":")[0] if first_leg else edfp
            first_amt  = first_leg.split(":")[1] if ":" in first_leg else ""
            changes_note = ""
            if scn and scn != "none":
                changes_note = " Some spending adjustments are needed."
            return (
                f"Use {n} installment{'s' if n != 1 else ''} of {cur} {first_amt}, "
                f"starting {first_date}. "
                f"This leaves at least {fmt(min_bal)} available.{changes_note}"
            )
        elif method == "partial_payment":
            remainder = round(req_amount - safe, 2)
            return (
                f"Pay {fmt(safe)} today and {fmt(remainder)} on {edfp}. "
                f"This spreads the cost while keeping at least {fmt(min_bal)} available."
            )
        elif method == "full_payment":
            changes_note = ""
            if scn and scn != "none":
                changes_note = f" Reduce or stop some flexible expenses first."
            return (
                f"Pay {fmt(req_amount)} today with spending adjustments.{changes_note} "
                f"This maintains at least {fmt(min_bal)} minimum balance."
            )
        else:
            return f"Use {method} plan to complete the request safely."

    elif status == "affordable_later":
        return (
            f"Wait until {edfp} to pay {fmt(req_amount)} in full. "
            f"Current safe amount is {fmt(safe)}. "
            f"Balance will be sufficient after upcoming income clears."
        )

    elif status == "not_affordable":
        return (
            f"Cannot safely commit {fmt(req_amount)} within the 90-day forecast. "
            f"Maximum safe amount today is {fmt(safe)}. "
            f"Consider reducing the request amount or revisiting later."
        )

    return f"Recommendation: {method}. Safe amount: {fmt(safe)}."
