from pathlib import Path

from tfm_sae_evals.config import ExternalPaths


def test_external_paths_missing(tmp_path):
    config = tmp_path / "paths.toml"
    config.write_text(
        "\n".join(
            [
                f'cub_raw_dir = "{tmp_path / "cub"}"',
                f'cub_processed_dir = "{tmp_path / "processed"}"',
                f'convnext_checkpoint = "{tmp_path / "model.pt"}"',
                f'cub_feature_cache = "{tmp_path / "features.pt"}"',
            ]
        ),
        encoding="utf-8",
    )
    paths = ExternalPaths.from_toml(config)
    assert set(paths.missing()) == {"cub_raw_dir", "cub_processed_dir", "convnext_checkpoint", "cub_feature_cache"}
