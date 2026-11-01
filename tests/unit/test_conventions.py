from datetime import UTC, date, datetime

import pytest

from vsm.conventions import (
    date_to_ns,
    datetime_to_ns,
    ms_to_ns,
    ns_to_datetime,
    ns_to_ms,
    parse_expiry_code,
    parse_option_symbol,
)


def test_ms_ns_round_trip() -> None:
    assert ms_to_ns(1_793_566_884_039) == 1_793_566_884_039_000_000
    assert ns_to_ms(ms_to_ns(1_793_566_884_039)) == 1_793_566_884_039


def test_datetime_ns_round_trip() -> None:
    ts = datetime(2026, 12, 1, 8, 15, 0, 250_000, tzinfo=UTC)
    assert ns_to_datetime(datetime_to_ns(ts)) == ts


def test_naive_datetime_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        datetime_to_ns(datetime(2026, 12, 1))


def test_date_to_ns_is_midnight_utc() -> None:
    assert ns_to_datetime(date_to_ns(date(2026, 11, 2))) == datetime(2026, 11, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("27NOV26", datetime(2026, 11, 27, 8, tzinfo=UTC)),
        ("2NOV26", datetime(2026, 11, 2, 8, tzinfo=UTC)),
    ],
)
def test_expiry_code_is_08_utc(code: str, expected: datetime) -> None:
    assert parse_expiry_code(code) == expected


def test_parse_option_symbol() -> None:
    expiry_ns, strike, cp = parse_option_symbol("BTC-27NOV26-70000-C")
    assert ns_to_datetime(expiry_ns) == datetime(2026, 11, 27, 8, tzinfo=UTC)
    assert strike == 70_000.0
    assert cp == 1
    assert parse_option_symbol("ETH-4NOV26-2360-P")[2] == -1


def test_parse_decimal_strike() -> None:
    assert parse_option_symbol("SOL-4NOV26-187d5-P")[1] == 187.5


@pytest.mark.parametrize("symbol", ["BTC-PERPETUAL", "BTC-29JAN27", "BTC-1NOV26-70000-X"])
def test_parse_rejects_non_options(symbol: str) -> None:
    with pytest.raises(ValueError, match="not an option"):
        parse_option_symbol(symbol)
