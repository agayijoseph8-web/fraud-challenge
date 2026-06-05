"""
Défi — Détection de fraude financière.

Vous devez implémenter la fonction `detect_fraud`.
La fonction `load_transactions` vous est FOURNIE (ne la modifiez pas).
"""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import atan2, cos, isfinite, radians, sin, sqrt
from statistics import median


def load_transactions(path):
    """Lit un fichier CSV de transactions et renvoie une liste de dicts."""
    transactions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transactions.append(_clean_row(row))
    return transactions


def _clean_row(row):
    def get(key):
        v = row.get(key)
        return v.strip() if isinstance(v, str) and v.strip() != "" else None

    amount_raw = get("amount")
    try:
        amount = float(amount_raw) if amount_raw is not None else None
    except ValueError:
        amount = None

    card_raw = get("card_present")
    if card_raw is None:
        card_present = None
    else:
        card_present = card_raw.lower() in ("true", "1", "yes", "oui")

    return {
        "transaction_id": get("transaction_id"),
        "timestamp": get("timestamp"),
        "user_id": get("user_id"),
        "amount": amount,
        "currency": get("currency"),
        "merchant": get("merchant"),
        "country": get("country"),
        "card_present": card_present,
    }


def detect_fraud(transactions):
    """Analyse une liste de transactions et renvoie un verdict pour chacune.

    Retour : list[dict] avec transaction_id, fraud_score (0-1),
    is_suspicious (bool), reason (str) — un résultat par transaction, même ordre.
    """
    try:
        rows = list(transactions or [])
    except TypeError:
        return []

    records = [_normalize_transaction(row, index) for index, row in enumerate(rows)]
    signals = defaultdict(list)

    id_counts = Counter(
        record["transaction_id"]
        for record in records
        if not _missing(record["transaction_id"])
    )
    signature_counts = Counter(
        record["duplicate_signature"]
        for record in records
        if record["duplicate_signature"] is not None
    )

    users = defaultdict(list)
    for record in records:
        users[record["user_key"]].append(record)

    for record in records:
        _score_basic_anomalies(record, id_counts, signature_counts, signals)
        _score_amount_anomaly(record, users[record["user_key"]], signals)

    for user_records in users.values():
        _score_fast_activity(user_records, signals)
        _score_impossible_travel(user_records, signals)
        _score_unusual_context(user_records, signals)

    results = []
    for record in records:
        record_signals = signals.get(record["index"], [])
        fraud_score = _combine_scores(score for score, _ in record_signals)
        is_suspicious = fraud_score >= 0.65
        reason = _build_reason(record_signals, is_suspicious)
        results.append(
            {
                "transaction_id": record["transaction_id"],
                "fraud_score": fraud_score,
                "is_suspicious": bool(is_suspicious),
                "reason": reason,
            }
        )

    return results


def _normalize_transaction(row, index):
    tx = row if isinstance(row, dict) else {}
    amount = _to_float(tx.get("amount"))
    timestamp = _parse_timestamp(tx.get("timestamp"))
    user_id = _clean_text(tx.get("user_id"))
    country = _clean_code(tx.get("country"))
    currency = _clean_code(tx.get("currency"))
    merchant = _clean_text(tx.get("merchant"))
    card_present = _to_bool(tx.get("card_present"))

    signature = None
    if user_id and timestamp and amount is not None and merchant:
        signature = (
            user_id,
            timestamp.isoformat(),
            round(amount, 2),
            merchant.lower(),
            currency,
            country,
        )

    return {
        "index": index,
        "raw": tx,
        "transaction_id": tx.get("transaction_id"),
        "timestamp": timestamp,
        "user_id": user_id,
        "user_key": user_id or f"__missing_user_{index}",
        "amount": amount,
        "currency": currency,
        "merchant": merchant,
        "country": country,
        "card_present": card_present,
        "duplicate_signature": signature,
    }


def _score_basic_anomalies(record, id_counts, signature_counts, signals):
    tx = record["raw"]
    missing_fields = []
    for field in (
        "transaction_id",
        "timestamp",
        "user_id",
        "amount",
        "currency",
        "merchant",
        "country",
        "card_present",
    ):
        if field == "amount":
            if record["amount"] is None:
                missing_fields.append(field)
        elif field == "timestamp":
            if record["timestamp"] is None:
                missing_fields.append(field)
        elif field == "card_present":
            if record["card_present"] is None:
                missing_fields.append(field)
        elif _missing(tx.get(field)):
            missing_fields.append(field)

    if record["amount"] is None:
        _add_signal(signals, record, 0.9, "Montant manquant ou invalide")
    elif record["amount"] <= 0:
        _add_signal(signals, record, 0.9, "Montant nul ou negatif")

    other_missing = [field for field in missing_fields if field != "amount"]
    if other_missing:
        score = 0.85 if any(field in other_missing for field in ("transaction_id", "user_id", "country")) else 0.7
        _add_signal(
            signals,
            record,
            score,
            "Champs obligatoires manquants: " + ", ".join(other_missing),
        )

    transaction_id = record["transaction_id"]
    if not _missing(transaction_id) and id_counts[transaction_id] > 1:
        _add_signal(signals, record, 0.9, "Identifiant de transaction duplique")

    signature = record["duplicate_signature"]
    if signature is not None and signature_counts[signature] > 1:
        _add_signal(signals, record, 0.78, "Transaction identique repetee")


def _score_amount_anomaly(record, user_records, signals):
    amount = record["amount"]
    if amount is None or amount <= 0:
        return

    reference = _reference_amounts(record, user_records)
    if not reference:
        if amount >= 10000 and record["card_present"] is False:
            _add_signal(signals, record, 0.66, "Montant eleve sans carte presente")
        return

    med = median(reference)
    if med <= 0:
        return

    deviations = [abs(value - med) for value in reference]
    mad = median(deviations) if deviations else 0.0
    ratio = amount / med
    gap = amount - med

    if len(reference) >= 3:
        if ratio >= 8 and gap >= max(100.0, med * 3):
            _add_signal(signals, record, 0.9, "Montant tres superieur a l'habitude du client")
        elif ratio >= 5 and gap >= max(100.0, med * 2):
            _add_signal(signals, record, 0.78, "Montant inhabituel pour ce client")
        elif ratio >= 3 and gap >= max(500.0, mad * 8, med * 2):
            _add_signal(signals, record, 0.68, "Montant au-dessus du profil habituel")
    elif ratio >= 15 and gap >= 1000.0:
        _add_signal(signals, record, 0.72, "Montant tres eleve par rapport aux achats connus")


def _score_fast_activity(user_records, signals):
    dated = sorted(
        (record for record in user_records if record["timestamp"] is not None),
        key=lambda record: record["timestamp"],
    )
    for record in dated:
        in_10_min = _count_nearby(dated, record, minutes=10)
        in_60_min = _count_nearby(dated, record, minutes=60)
        if in_10_min >= 4:
            _add_signal(signals, record, 0.8, "Trop de transactions en quelques minutes")
        elif in_60_min >= 6:
            _add_signal(signals, record, 0.72, "Frequence de transactions anormale")


def _score_impossible_travel(user_records, signals):
    dated = sorted(
        (
            record
            for record in user_records
            if record["timestamp"] is not None and record["country"]
        ),
        key=lambda record: record["timestamp"],
    )
    for previous, current in zip(dated, dated[1:]):
        if previous["country"] == current["country"]:
            continue

        seconds = abs((current["timestamp"] - previous["timestamp"]).total_seconds())
        if seconds <= 0:
            seconds = 1
        hours = seconds / 3600
        distance = _country_distance_km(previous["country"], current["country"])

        suspicious = False
        if distance is None:
            suspicious = hours <= 2
        else:
            required_speed = distance / hours
            suspicious = (
                (distance >= 1000 and hours <= 24 and required_speed > 700)
                or (distance >= 500 and hours <= 3)
                or (distance >= 300 and hours <= 1)
            )

        if suspicious:
            reason = "Deux pays differents en trop peu de temps"
            _add_signal(signals, previous, 0.88, reason)
            _add_signal(signals, current, 0.88, reason)


def _score_unusual_context(user_records, signals):
    countries = [record["country"] for record in user_records if record["country"]]
    currencies = [record["currency"] for record in user_records if record["currency"]]
    if len(countries) < 4 and len(currencies) < 4:
        return

    common_country = Counter(countries).most_common(1)[0][0] if countries else None
    common_currency = Counter(currencies).most_common(1)[0][0] if currencies else None
    for record in user_records:
        context_changed = (
            (common_country and record["country"] and record["country"] != common_country)
            or (common_currency and record["currency"] and record["currency"] != common_currency)
        )
        if context_changed and record["card_present"] is False:
            _add_signal(signals, record, 0.45, "Contexte inhabituel sans carte presente")


def _reference_amounts(record, user_records):
    current_index = record["index"]
    valid = [
        other
        for other in user_records
        if other["index"] != current_index
        and other["amount"] is not None
        and other["amount"] > 0
    ]
    if not valid:
        return []

    timestamp = record["timestamp"]
    if timestamp is not None:
        previous = [
            other["amount"]
            for other in valid
            if other["timestamp"] is not None and other["timestamp"] < timestamp
        ]
        if len(previous) >= 2:
            return previous

    return [other["amount"] for other in valid]


def _count_nearby(dated_records, target, minutes):
    window = minutes * 60
    target_ts = target["timestamp"]
    return sum(
        1
        for record in dated_records
        if abs((record["timestamp"] - target_ts).total_seconds()) <= window
    )


def _add_signal(signals, record, score, reason):
    signals[record["index"]].append((float(score), reason))


def _combine_scores(scores):
    probability_clean = 1.0
    for score in scores:
        bounded = max(0.0, min(1.0, float(score)))
        probability_clean *= 1.0 - bounded
    return round(max(0.0, min(1.0, 1.0 - probability_clean)), 2)


def _build_reason(signals, is_suspicious):
    if not signals:
        return "Transaction conforme au profil du client"
    if not is_suspicious:
        return "Signal faible, transaction conservee comme normale"

    ordered = sorted(signals, key=lambda item: item[0], reverse=True)
    reasons = []
    for _, reason in ordered:
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) == 2:
            break
    return "; ".join(reasons)


def _missing(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def _clean_text(value):
    if _missing(value):
        return None
    return str(value).strip()


def _clean_code(value):
    text = _clean_text(value)
    return text.upper() if text else None


def _to_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if isfinite(number) else None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if _missing(value):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "oui"):
            return True
        if text in ("false", "0", "no", "non"):
            return False
    return None


def _parse_timestamp(value):
    if _missing(value):
        return None
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _country_distance_km(country_a, country_b):
    coords_a = _COUNTRY_COORDS.get(country_a)
    coords_b = _COUNTRY_COORDS.get(country_b)
    if coords_a is None or coords_b is None:
        return None

    lat1, lon1 = coords_a
    lat2, lon2 = coords_b
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * radius * atan2(sqrt(a), sqrt(1 - a))


_COUNTRY_COORDS = {
    "AE": (23.4, 53.8),
    "AU": (-25.3, 133.8),
    "BE": (50.5, 4.5),
    "BF": (12.2, -1.6),
    "BJ": (9.3, 2.3),
    "BR": (-14.2, -51.9),
    "CA": (56.1, -106.3),
    "CH": (46.8, 8.2),
    "CI": (7.5, -5.5),
    "CM": (7.4, 12.4),
    "CN": (35.9, 104.2),
    "DE": (51.2, 10.5),
    "DZ": (28.0, 1.7),
    "EG": (26.8, 30.8),
    "ES": (40.5, -3.7),
    "FR": (46.2, 2.2),
    "GA": (-0.8, 11.6),
    "GB": (55.4, -3.4),
    "GH": (7.9, -1.0),
    "IN": (20.6, 78.9),
    "IT": (41.9, 12.6),
    "JP": (36.2, 138.3),
    "MA": (31.8, -7.1),
    "ML": (17.6, -3.9),
    "NE": (17.6, 8.1),
    "NG": (9.1, 8.7),
    "NL": (52.1, 5.3),
    "RU": (61.5, 105.3),
    "SN": (14.5, -14.5),
    "TG": (8.6, 1.0),
    "TN": (33.9, 9.5),
    "TR": (38.9, 35.2),
    "US": (37.1, -95.7),
    "ZA": (-30.6, 22.9),
}
