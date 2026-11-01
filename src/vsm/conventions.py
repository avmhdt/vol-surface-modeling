"""Unit and time conventions. Every venue-unit conversion in the project lives here."""

from datetime import UTC, date, datetime

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000

EXPIRY_HOUR_UTC = 8

_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def ms_to_ns(ms: int) -> int:
    """Venue millisecond timestamp to int64 nanoseconds UTC."""
    return int(ms) * NS_PER_MS


def ns_to_ms(ns: int) -> int:
    """Int64 nanoseconds UTC to venue milliseconds, truncating."""
    return int(ns) // NS_PER_MS


def datetime_to_ns(ts: datetime) -> int:
    """Timezone-aware datetime to int64 nanoseconds UTC."""
    if ts.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(ts.timestamp()) * NS_PER_S + ts.microsecond * 1_000


def ns_to_datetime(ns: int) -> datetime:
    """Int64 nanoseconds UTC to an aware datetime, truncated to microseconds."""
    seconds, remainder = divmod(int(ns), NS_PER_S)
    return datetime.fromtimestamp(seconds, UTC).replace(microsecond=remainder // 1_000)


def date_to_ns(day: date) -> int:
    """Midnight UTC of a calendar date in nanoseconds."""
    return datetime_to_ns(datetime(day.year, day.month, day.day, tzinfo=UTC))


def parse_expiry_code(code: str) -> datetime:
    """Deribit expiry code such as ``27NOV26`` to 08:00 UTC on that date."""
    day, month, year = code[:-5], code[-5:-2], code[-2:]
    return datetime(2000 + int(year), _MONTHS[month], int(day), EXPIRY_HOUR_UTC, tzinfo=UTC)


def parse_option_symbol(symbol: str) -> tuple[int, float, int]:
    """``BTC-27NOV26-70000-C`` to (expiry_ns, strike USD, cp), cp +1 call or -1 put."""
    parts = symbol.split("-")
    if len(parts) != 4 or parts[3] not in ("C", "P"):
        raise ValueError(f"not an option symbol: {symbol}")
    strike = float(parts[2].replace("d", "."))
    return datetime_to_ns(parse_expiry_code(parts[1])), strike, 1 if parts[3] == "C" else -1
