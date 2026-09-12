"""Configuration helpers for external data and artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExternalPaths:
    cub_raw_dir: Path
    cub_processed_dir: Path
    convnext_checkpoint: Path
    cub_feature_cache: Path

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExternalPaths":
        path = Path(path)
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return cls(
            cub_raw_dir=Path(data["cub_raw_dir"]).expanduser(),
            cub_processed_dir=Path(data["cub_processed_dir"]).expanduser(),
            convnext_checkpoint=Path(data["convnext_checkpoint"]).expanduser(),
            cub_feature_cache=Path(data["cub_feature_cache"]).expanduser(),
        )

    def missing(self) -> dict[str, Path]:
        fields = {
            "cub_raw_dir": self.cub_raw_dir,
            "cub_processed_dir": self.cub_processed_dir,
            "convnext_checkpoint": self.convnext_checkpoint,
            "cub_feature_cache": self.cub_feature_cache,
        }
        return {name: path for name, path in fields.items() if not path.exists()}
