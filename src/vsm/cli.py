"""The ``vsm`` command line."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import jsonschema
import typer

from vsm.config import AppConfig, config_hash, load_config
from vsm.conventions import datetime_to_ns
from vsm.explog import append_run
from vsm.export.reader import read_surface_bundle
from vsm.io import recorder
from vsm.io.deribit import DeribitClient
from vsm.io.history import ingest_history
from vsm.io.smoke import run_smoke
from vsm.logs import configure_logging
from vsm.pipeline import ALL_STEPS, export_day, recheck_day, refresh_outputs, run_day
from vsm.settings import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Volatility surface pipeline.")
ingest_app = typer.Typer(
    no_args_is_help=True, help="Fetch metadata and history into the raw store."
)
export_app = typer.Typer(help="Write and validate vsm.surface.v1 bundles.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(export_app, name="export")

ConfigOption = Annotated[Path | None, typer.Option("--config", help="YAML config path.")]
CurrencyOption = Annotated[str, typer.Option("--currency", help="Comma-separated currencies.")]


def _split(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def _client(settings: Settings, config: AppConfig) -> DeribitClient:
    return DeribitClient(
        settings.deribit_base_url,
        max_rps=config.recorder.max_requests_per_second,
        retries=config.recorder.retries,
    )


@app.command()
def smoke(currency: CurrencyOption = "BTC,ETH") -> None:
    """Call each Deribit public endpoint once; nonzero exit if any check fails."""
    settings = Settings()
    config = load_config()
    configure_logging(settings.log_level, as_json=settings.log_json)
    with _client(settings, config) as client:
        results = run_smoke(client, _split(currency))
    width = max(len(r.endpoint) for r in results)
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        typer.echo(f"{mark}  {result.endpoint:<{width}}  {result.detail}")
    failed = sum(not r.ok for r in results)
    typer.echo(f"{len(results) - failed}/{len(results)} checks passed")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def record(
    currency: Annotated[str | None, typer.Option("--currency")] = None,
    kinds: Annotated[str | None, typer.Option("--kinds")] = None,
    ticker_sweep: Annotated[bool | None, typer.Option("--ticker-sweep/--no-ticker-sweep")] = None,
    config_path: ConfigOption = None,
) -> None:
    """Take one snapshot of the configured currencies. Scheduled hourly at hh:15 UTC."""
    settings = Settings()
    config = load_config(config_path)
    cfg = config.recorder
    now = datetime.now(UTC)
    log_file = settings.data_dir / "logs" / f"recorder-{now:%Y-%m-%d}.jsonl"
    configure_logging(settings.log_level, as_json=settings.log_json, jsonl=log_file)
    slot = recorder.current_slot(now, cfg.minute_utc)
    sweep = ticker_sweep
    if sweep is None:
        sweep = recorder.wants_ticker_sweep(cfg.ticker_sweep, slot, cfg.canonical_time_utc)
    with _client(settings, config) as client:
        summary = recorder.record_snapshot(
            client,
            settings.data_dir,
            currencies=_split(currency) if currency else settings.currencies,
            kinds=_split(kinds) if kinds else list(cfg.kinds),
            minute_utc=cfg.minute_utc,
            ticker_sweep=sweep,
            now=now,
        )
    for o in summary.outcomes:
        typer.echo(f"{o.status:<8} {o.dataset:<14} {o.snapshot_id}  rows={o.rows} {o.reason}")
    typer.echo(
        f"slot {summary.slot.label} on_schedule={summary.slot.on_schedule} "
        f"tickers={summary.ticker_rows} retries={summary.retries}"
    )
    if not summary.ok:
        raise typer.Exit(code=1)


@app.command()
def status(
    since: Annotated[str | None, typer.Option("--since", help="YYYY-MM-DD")] = None,
    config_path: ConfigOption = None,
) -> None:
    """Recorder coverage per underlying; writes results/metrics/recorder_coverage.json."""
    settings = Settings()
    config = load_config(config_path)
    cfg = config.recorder
    now = datetime.now(UTC)
    start = date.fromisoformat(since) if since else None
    data_dir = settings.data_dir
    report = {
        "as_of": now.strftime("%Y-%m-%dT%H:%MZ"),
        "underlyings": {
            underlying: {
                **recorder.coverage(
                    data_dir,
                    underlying,
                    minute_utc=cfg.minute_utc,
                    canonical_time_utc=cfg.canonical_time_utc,
                    since=start,
                    now=now,
                ),
                "consecutive_hourly": recorder.consecutive_hourly(
                    data_dir, underlying, cfg.minute_utc
                ),
            }
            for underlying in cfg.currencies
        },
        "gaps_logged": len(recorder.read_gaps(data_dir)),
        "raw_mb": recorder.disk_use_mb(data_dir),
        "fs_used_pct": round(100 * recorder.check_disk(data_dir), 1),
    }
    out = config.results_dir() / "metrics" / "recorder_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command()
def build(
    underlying: Annotated[str, typer.Option("--underlying")],
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")],
    steps: Annotated[str, typer.Option("--steps")] = ",".join(ALL_STEPS),
    at: Annotated[str | None, typer.Option("--at", help="HH:MM UTC, default canonical.")] = None,
    config_path: ConfigOption = None,
) -> None:
    """Build every as-of date in a range. A date with no snapshot is logged and skipped."""
    settings = Settings()
    config = load_config(config_path)
    configure_logging(settings.log_level, as_json=settings.log_json)
    chosen = tuple(_split(steps))
    unknown = set(chosen) - set(ALL_STEPS)
    if unknown:
        typer.echo(f"unknown steps: {sorted(unknown)}")
        raise typer.Exit(code=2)
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    built = 0
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        report = run_day(settings.data_dir, config, underlying, day, at_utc=at, steps=chosen)
        if report is None:
            typer.echo(f"{day}: no snapshot")
            continue
        built += 1
        check = report["mark_iv_check"]
        typer.echo(
            f"{day}: {report['snapshot']} rows={report['rows']} kept={report['kept']} "
            f"liquid={report['liquid']} forwards={report['forward_methods']} "
            f"mark_iv_median={check['median_abs_diff_vol_pts']} "
            f"surface={report.get('surface')}"
        )
    metrics = refresh_outputs(settings.data_dir, config, underlying)
    typer.echo(f"metrics: {metrics}")
    append_run(
        config.results_dir(),
        "build",
        config_hash(config),
        underlying=underlying,
        start=start,
        end=end,
        params={"steps": _split(steps), "at": at},
        metrics={"days_built": built, **metrics},
    )
    if built == 0:
        raise typer.Exit(code=1)


def _days(start: str, end: str) -> list[date]:
    first = date.fromisoformat(start)
    span = (date.fromisoformat(end) - first).days + 1
    return [first + timedelta(days=offset) for offset in range(span)]


@app.command()
def check(
    underlying: Annotated[str, typer.Option("--underlying")],
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")],
    config_path: ConfigOption = None,
) -> None:
    """Re-run the arbitrage gate on stored surfaces, without refitting."""
    settings = Settings()
    config = load_config(config_path)
    configure_logging(settings.log_level, as_json=settings.log_json)
    failures = 0
    for day in _days(start, end):
        report = recheck_day(settings.data_dir, config, underlying, day)
        if report is None:
            typer.echo(f"{day}: no stored surface")
            continue
        failures += not report.passed
        typer.echo(
            f"{day}: passed={report.passed} min_g={report.min_g:+.3e} "
            f"min_calendar_dw={report.min_calendar_dw:+.3e} "
            f"density_mass_max_err={report.density_mass_max_err:.2e}"
        )
    if failures:
        raise typer.Exit(code=1)


@export_app.callback(invoke_without_command=True)
def export(
    ctx: typer.Context,
    underlying: Annotated[str | None, typer.Option("--underlying")] = None,
    day: Annotated[str | None, typer.Option("--date")] = None,
    start: Annotated[str | None, typer.Option("--start")] = None,
    end: Annotated[str | None, typer.Option("--end")] = None,
) -> None:
    """Write bundles of stored surfaces for a date or range."""
    if ctx.invoked_subcommand is not None:
        return
    first = day or start
    if underlying is None or first is None:
        typer.echo("give --underlying and either --date or --start and --end")
        raise typer.Exit(code=2)
    settings = Settings()
    config = load_config()
    configure_logging(settings.log_level, as_json=settings.log_json)
    for current in _days(first, end or first):
        path = export_day(settings.data_dir, config, underlying, current)
        typer.echo(f"{current}: {path or 'no stored surface'}")


@export_app.command("validate")
def export_validate(path: Annotated[str, typer.Argument(help="Path to a surface.json.")]) -> None:
    """Validate one file against the schema and the consumer invariants."""
    try:
        bundle = read_surface_bundle(path)
    except (ValueError, jsonschema.ValidationError) as exc:
        typer.echo(f"invalid: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"valid: {bundle.document['underlying']} {bundle.document['as_of']}")


@ingest_app.command("instruments")
def ingest_instruments(
    currency: CurrencyOption = "BTC,ETH",
    expired: Annotated[bool, typer.Option("--expired", help="Also fetch expired.")] = False,
) -> None:
    """Merge active (and recently expired) instrument metadata into the stored table."""
    settings = Settings()
    config = load_config()
    configure_logging(settings.log_level, as_json=settings.log_json)
    seen_ns = datetime_to_ns(datetime.now(UTC))
    with _client(settings, config) as client:
        for underlying in _split(currency):
            table = recorder.refresh_instruments(
                client,
                settings.data_dir,
                underlying,
                ["option", "future"],
                seen_ns,
                expired=expired,
            )
            typer.echo(f"{underlying}: {table.height} instruments stored")


@ingest_app.command("deribit-history")
def ingest_deribit_history(
    what: Annotated[str, typer.Option("--what")] = "dvol,perp,funding,delivery",
    currency: CurrencyOption = "BTC,ETH",
    start: Annotated[str | None, typer.Option("--start", help="Default: resume.")] = None,
    end: Annotated[str | None, typer.Option("--end", help="Default: now.")] = None,
) -> None:
    """DVOL, perpetual OHLC, funding and delivery prices: backfill, then daily append."""
    settings = Settings()
    config = load_config()
    configure_logging(settings.log_level, as_json=settings.log_json)
    with _client(settings, config) as client:
        counts = ingest_history(
            client,
            settings.data_dir,
            what=_split(what),
            currencies=_split(currency),
            start=date.fromisoformat(start) if start else None,
            end=date.fromisoformat(end) if end else None,
        )
    for series, rows in counts.items():
        typer.echo(f"{series}: {rows} rows")


if __name__ == "__main__":
    app()
