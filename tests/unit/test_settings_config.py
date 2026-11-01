from pathlib import Path

import pytest
import yaml

from vsm.config import config_hash, load_config
from vsm.settings import Settings


def clean_settings(**env: str) -> Settings:
    return Settings(_env_file=None, **env)  # type: ignore[call-arg]


def test_data_dir_defaults_to_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VSM_DATA_DIR", raising=False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    settings = clean_settings()
    assert settings.data_dir == Path("~/data/vol-surface-modeling").expanduser()


def test_data_dir_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", str(tmp_path))
    assert clean_settings().data_dir == tmp_path


def test_empty_data_dir_means_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", "")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert clean_settings().data_dir == tmp_path


def test_mnt_drives_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSM_DATA_DIR", "/mnt/d/data")
    with pytest.raises(ValueError, match="/mnt"):
        _ = clean_settings().data_dir


def test_currencies_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VSM_RECORDER_CURRENCIES", "BTC, ETH")
    assert clean_settings().currencies == ["BTC", "ETH"]


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARDIS_API_KEY", "not-a-real-key")
    assert "not-a-real-key" not in repr(clean_settings())


def test_default_config_loads_with_spec_values() -> None:
    config = load_config()
    assert config.recorder.minute_utc == 15
    assert config.recorder.canonical_time_utc == "08:15"
    assert config.recorder.retries == 5
    assert config.results_dir().name == "results"
    assert config.reports_dir().name == "reports"


def test_config_hash_ignores_formatting_not_values(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    same = tmp_path / "same.yaml"
    same.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    raw["recorder"]["minute_utc"] = 20
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert config_hash(load_config(same)) == config_hash(load_config())
    assert config_hash(load_config(other)) != config_hash(load_config())
    assert config_hash(load_config()).startswith("sha256:")
