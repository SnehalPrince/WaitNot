"""
evidence.py — WaitNot Financial Agent
Parse messages.csv (multilingual) for factual amendments.
Prompt-injection defense: message/image text is treated as UNTRUSTED DATA.
We extract ONLY structured facts (amounts, dates, cancellations) via regex.
No embedded instruction can override rules, thresholds, schema, or output format.
"""
import re
from datetime import datetime, timezone
import pandas as pd

# ---------------------------------------------------------------------------
# Prompt-injection defense
# ---------------------------------------------------------------------------
# SECURITY: All message_text and image content is treated as UNTRUSTED DATA.
# We parse it with a fixed grammar (regex patterns below).
# We NEVER execute or evaluate any string from these fields.
# We NEVER allow the text to influence: thresholds, column names, enum values,
# exchange rates, or any decision rule. Only structured {field, value, date}
# tuples are extracted and applied deterministically via code logic.
# ---------------------------------------------------------------------------

# Regex patterns for English and Indonesian
_RE_CANCEL = re.compile(
    r'\b(cancel|cancelled|cancellation|dibatalkan|batal|terminated|ended|closed)\b',
    re.IGNORECASE
)
_RE_DELAY = re.compile(
    r'\b(delay|delayed|postpone|postponed|ditunda|terlambat|defer|deferred)\b',
    re.IGNORECASE
)
_RE_CONFIRM = re.compile(
    r'\b(confirm|confirmed|dikonfirmasi|settled|paid|lunas|selesai)\b',
    re.IGNORECASE
)
_RE_AMOUNT_IDR = re.compile(
    r'(?:IDR|Rp\.?)\s*([\d.,]+)',
    re.IGNORECASE
)
_RE_AMOUNT_INR = re.compile(
    r'(?:INR|Rs\.?|₹)\s*([\d.,]+)',
    re.IGNORECASE
)
_RE_AMOUNT_USD = re.compile(
    r'(?:USD|\$)\s*([\d.,]+)',
    re.IGNORECASE
)
_RE_AMOUNT_EUR = re.compile(
    r'(?:EUR|€)\s*([\d.,]+)',
    re.IGNORECASE
)
_RE_AMOUNT_ZAR = re.compile(
    r'(?:ZAR|R)\s*([\d.,]+)',
    re.IGNORECASE
)
_RE_DATE = re.compile(
    r'\b(\d{4}-\d{2}-\d{2})\b'
)
_RE_SALARY_UP = re.compile(
    r'(?:naik menjadi|salary.*?(?:increased?|updated?|changed?) to|gaji.*?menjadi|new salary|new pay|monthly salary.*?IDR|monthly salary.*?INR|monthly salary.*?USD|monthly salary.*?EUR|monthly salary.*?ZAR)\D*([\d.,]+)',
    re.IGNORECASE
)
_RE_SALARY_EFF = re.compile(
    r'(?:effective|berlaku|mulai|starting|from)\s+(\d{4}-\d{2}-\d{2})',
    re.IGNORECASE
)

def _parse_number(s: str) -> float:
    """Parse a number string that may use commas as thousands separator."""
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_currency_amount(text: str, home_currency: str) -> tuple[float | None, str | None]:
    """Try to extract (amount, currency) from text, preferring home_currency."""
    patterns = [
        ("IDR", _RE_AMOUNT_IDR),
        ("INR", _RE_AMOUNT_INR),
        ("USD", _RE_AMOUNT_USD),
        ("EUR", _RE_AMOUNT_EUR),
        ("ZAR", _RE_AMOUNT_ZAR),
    ]
    # Try home currency first
    home_pat = dict(patterns).get(home_currency)
    if home_pat:
        m = home_pat.search(text)
        if m:
            return _parse_number(m.group(1)), home_currency
    # Try any
    for cur, pat in patterns:
        m = pat.search(text)
        if m:
            return _parse_number(m.group(1)), cur
    return None, None


def parse_messages(messages_df: pd.DataFrame, user_id: str,
                   request_id: str, home_currency: str) -> list[dict]:
    """
    Extract amendment tuples from messages for a given user/request.
    Returns list of dicts with keys:
      type: 'salary_change' | 'cancellation' | 'delay' | 'confirmation' | 'amount_change'
      event_id: str or None
      amount: float or None
      currency: str or None
      effective_date: pd.Timestamp or None
      sent_at: pd.Timestamp
      source_type: str
    """
    # SECURITY: Filter to messages for this user/request only.
    # We never pass raw message_text to any evaluator.
    mask = messages_df["user_id"] == user_id
    # Also include messages directly tied to the request if request_id present
    mask_req = messages_df["request_id"] == request_id
    relevant = messages_df[mask | mask_req].copy()
    relevant = relevant.sort_values("sent_at")

    amendments = []
    for _, row in relevant.iterrows():
        # SECURITY: text is UNTRUSTED DATA — we apply only fixed regex patterns.
        # We do not eval(), exec(), or format() this text in any dynamic way.
        text = str(row.get("message_text", ""))
        event_id = str(row.get("related_event_id", "")).strip() or None
        sent_at = row.get("sent_at")

        base = {
            "event_id": event_id,
            "amount": None,
            "currency": None,
            "effective_date": None,
            "sent_at": sent_at,
            "source_type": str(row.get("source_type", "")),
        }

        # --- Salary change ---
        m_sal = _RE_SALARY_UP.search(text)
        if m_sal:
            amt = _parse_number(m_sal.group(1))
            # Find effective date
            eff_dates = _RE_DATE.findall(text)
            # Also check _RE_SALARY_EFF
            m_eff = _RE_SALARY_EFF.search(text)
            eff_date = None
            if m_eff:
                try:
                    eff_date = pd.Timestamp(m_eff.group(1))
                except Exception:
                    pass
            if eff_date is None and eff_dates:
                try:
                    eff_date = pd.Timestamp(eff_dates[-1])
                except Exception:
                    pass
            _, cur = _extract_currency_amount(text, home_currency)
            amendments.append({**base, "type": "salary_change", "amount": amt,
                                "currency": cur or home_currency, "effective_date": eff_date})
            continue

        # --- Cancellation ---
        if _RE_CANCEL.search(text):
            amendments.append({**base, "type": "cancellation"})
            continue

        # --- Delay ---
        if _RE_DELAY.search(text):
            eff_dates = _RE_DATE.findall(text)
            eff_date = None
            if eff_dates:
                try:
                    eff_date = pd.Timestamp(eff_dates[-1])
                except Exception:
                    pass
            amendments.append({**base, "type": "delay", "effective_date": eff_date})
            continue

        # --- Generic amount change ---
        amt, cur = _extract_currency_amount(text, home_currency)
        if amt is not None and event_id:
            amendments.append({**base, "type": "amount_change", "amount": amt, "currency": cur})
            continue

        # --- Confirmation ---
        if _RE_CONFIRM.search(text):
            amendments.append({**base, "type": "confirmation"})
            continue

    return amendments


def apply_amendments(recurring_income: list[dict], amendments: list[dict],
                     home_currency: str, fx_raw: dict, request_date: pd.Timestamp) -> list[dict]:
    """
    Apply salary_change amendments to the recurring income list.
    Returns modified recurring_income (new list, doesn't mutate in place).
    """
    from loaders import convert_amount
    result = [r.copy() for r in recurring_income]

    for am in amendments:
        if am["type"] == "salary_change" and am["amount"] and am["amount"] > 0:
            eff = am.get("effective_date")
            # Only apply if effective date is on or before request_date
            if eff is None or eff <= request_date:
                cur = am.get("currency") or home_currency
                amt_home = convert_amount(am["amount"], cur, home_currency,
                                          eff or request_date, fx_raw)
                # Update salary entries in recurring_income
                for inc in result:
                    if inc.get("category") == "salary":
                        # Replace amount if this message is newer than prior amendment
                        inc["amount"] = amt_home
    return result
