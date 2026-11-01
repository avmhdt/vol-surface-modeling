"""Environment settings (.env). Secrets are SecretStr and are never logged."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process settings read from the environment and the repo's ``.env``."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    data_root: Path = Field(default=Path("~/data/vol-surface-modeling"), alias="DATA_ROOT")
    vsm_data_dir: Path | None = Field(default=None, alias="VSM_DATA_DIR")
    deribit_base_url: str = Field(
        default="https://www.deribit.com/api/v2", alias="DERIBIT_BASE_URL"
    )
    log_level: str = Field(default="INFO", alias="VSM_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="VSM_LOG_JSON")
    recorder_currencies: str = Field(default="BTC,ETH", alias="VSM_RECORDER_CURRENCIES")
    tardis_datasets_url: str = Field(
        default="https://datasets.tardis.dev/v1", alias="TARDIS_DATASETS_URL"
    )
    tardis_api_key: SecretStr | None = Field(default=None, alias="TARDIS_API_KEY")
    fred_api_key: SecretStr | None = Field(default=None, alias="FRED_API_KEY")
    wrds_username: str | None = Field(default=None, alias="WRDS_USERNAME")
    thetadata_base_url: str | None = Field(default=None, alias="THETADATA_BASE_URL")

    @property
    def data_dir(self) -> Path:
        """Root of the data tree. Refuses ``/mnt/*``: data must stay on WSL ext4."""
        root = (self.vsm_data_dir or self.data_root).expanduser()
        if root.as_posix().startswith("/mnt/"):
            raise ValueError(f"data dir {root} is on a /mnt drive; use the WSL ext4 filesystem")
        return root

    @property
    def currencies(self) -> list[str]:
        """Currencies the recorder snapshots by default."""
        return [c.strip() for c in self.recorder_currencies.split(",") if c.strip()]
