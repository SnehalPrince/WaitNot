"""
validate.py — WaitNot Financial Agent
Schema and business-rule validation of output rows before writing output.csv.
Fails loudly (raises ValueError) rather than writing an invalid file.
"""
import re

VALID_STATUS  = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}

_DATE_RE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLAN_LEG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:\d+\.?\d*$")
_CHANGE_RE   = re.compile(r"^(stop:[a-z0-9_]+|reduce_to:[a-z0-9_]+:\d+\.?\d*)$")


def validate_row(row: dict, requested_amount: float) -> None:
    """
    Validate a single output row. Raises ValueError with a clear message on failure.
    """
    rid = row.get("request_id", "?")

    # --- amount_safe_to_pay ---
    safe = row.get("amount_safe_to_pay")
    try:
        safe = float(safe)
    except (TypeError, ValueError):
        raise ValueError(f"{rid}: amount_safe_to_pay is not numeric: {safe!r}")
    if not (0 <= safe <= requested_amount + 0.01):  # +0.01 for fp rounding
        raise ValueError(
            f"{rid}: amount_safe_to_pay {safe} out of range [0, {requested_amount}]"
        )

    # --- affordability_status ---
    status = row.get("affordability_status", "")
    if status not in VALID_STATUS:
        raise ValueError(f"{rid}: invalid affordability_status {status!r}")

    # --- recommended_payment_method ---
    method = row.get("recommended_payment_method", "")
    if method not in VALID_METHODS:
        raise ValueError(f"{rid}: invalid recommended_payment_method {method!r}")

    # --- Status / method consistency ---
    if status == "affordable_now" and method != "full_payment":
        raise ValueError(f"{rid}: affordable_now requires full_payment, got {method!r}")
    if status == "not_affordable" and method != "not_recommended":
        raise ValueError(f"{rid}: not_affordable requires not_recommended, got {method!r}")
    if status == "affordable_later" and method != "wait":
        raise ValueError(f"{rid}: affordable_later requires wait, got {method!r}")

    # --- payment_plan ---
    plan = row.get("payment_plan", "none")
    if plan != "none":
        legs = plan.split("|")
        for leg in legs:
            if not _PLAN_LEG_RE.match(leg):
                raise ValueError(f"{rid}: invalid payment_plan leg {leg!r}")
        # partial_payment: must have exactly 2 legs summing to requested_amount
        if method == "partial_payment":
            if len(legs) != 2:
                raise ValueError(f"{rid}: partial_payment must have exactly 2 legs, got {len(legs)}")
            amounts = [float(l.split(":")[1]) for l in legs]
            total = round(sum(amounts), 2)
            if abs(total - requested_amount) > 0.05:
                raise ValueError(
                    f"{rid}: partial_payment legs sum {total} != requested_amount {requested_amount}"
                )
    elif method not in {"not_recommended"}:
        # Allowing plan=none for wait and affordable_later is actually valid
        # (wait means "don't pay now", full_payment today → plan has today's date)
        pass  # no strict requirement for plan=none on wait

    # --- earliest_date_for_full_payment ---
    edfp = row.get("earliest_date_for_full_payment", "")
    if status == "affordable_now":
        if not edfp or not _DATE_RE.match(str(edfp)):
            raise ValueError(f"{rid}: affordable_now requires earliest_date_for_full_payment = request_date")

    # --- spending_changes_needed ---
    scn = row.get("spending_changes_needed", "none")
    if scn != "none":
        changes = scn.split("|")
        if len(changes) > 3:
            raise ValueError(f"{rid}: spending_changes_needed has >3 entries")
        event_ids_seen = {}
        for ch in changes:
            if not _CHANGE_RE.match(ch):
                raise ValueError(f"{rid}: invalid spending_changes entry {ch!r}")
            parts = ch.split(":")
            action = parts[0]
            eid = parts[1]
            if eid in event_ids_seen:
                if event_ids_seen[eid] != action:
                    raise ValueError(
                        f"{rid}: event {eid} appears with both stop and reduce_to — mutually exclusive"
                    )
            event_ids_seen[eid] = action

    # --- decision_explanation ---
    expl = row.get("decision_explanation", "")
    if not str(expl).strip():
        raise ValueError(f"{rid}: decision_explanation is empty")


def validate_output(rows: list[dict], requests_df) -> None:
    """
    Validate all output rows. Raises ValueError on the first failure.
    """
    req_ids_expected = set(requests_df["request_id"].tolist())
    req_amounts      = dict(zip(requests_df["request_id"], requests_df["requested_amount"]))
    req_ids_got      = set()

    for row in rows:
        rid = row.get("request_id", "")
        req_ids_got.add(rid)
        req_amount = float(req_amounts.get(rid, 0))
        validate_row(row, req_amount)

    # Check completeness
    missing = req_ids_expected - req_ids_got
    if missing:
        raise ValueError(f"Missing output rows for request_ids: {sorted(missing)[:10]}")

    extra = req_ids_got - req_ids_expected
    if extra:
        raise ValueError(f"Extra output rows not in requests.csv: {sorted(extra)[:10]}")

    if len(rows) != len(req_ids_expected):
        raise ValueError(
            f"Row count mismatch: got {len(rows)}, expected {len(req_ids_expected)}"
        )

    print(f"[validate] All {len(rows)} rows passed validation.")
