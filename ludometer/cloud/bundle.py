"""The source bundle a job runs: the package, its lock, the configs. Nothing else.

``git ls-files`` limited to the paths the generator imports, so a bundle is a
few hundred kilobytes and never carries runs/, data/ or the web player.
Uncommitted edits ARE included (working tree, like microduck's helper), because
the whole point of a smoke job is to test what is on disk right now.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import tarfile
import tempfile
from pathlib import Path

__all__ = ["build_bundle", "upload_bundle"]

INCLUDE = ("ludometer", "pyproject.toml", "uv.lock", "configs", "README.md")
EXCLUDE_PARTS = ("__pycache__", ".pytest_cache")


def _repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return Path(out.decode().strip())


def build_bundle(out_dir: Path | None = None) -> tuple[Path, str]:
    """Write ``bundle-<sha>-<stamp>.tar.gz``; returns (path, short sha)."""
    root = _repo_root()
    sha = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        .decode()
        .strip()
    )
    files = (
        subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", *INCLUDE], cwd=root
        )
        .decode()
        .splitlines()
    )
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir or tempfile.mkdtemp())
    path = out_dir / f"bundle-{sha}-{stamp}.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for rel in files:
            if any(part in EXCLUDE_PARTS for part in Path(rel).parts):
                continue
            p = root / rel
            if p.is_file():
                tar.add(p, arcname=rel)
    return path, sha


def upload_bundle(repo_id: str, token: str | None = None) -> str:
    """Build and upload; returns the file name inside the repo."""
    from ludometer.cloud.hub import HfHub

    path, _sha = build_bundle()
    hub = HfHub(repo_id, repo_type="dataset", token=token)
    hub.put(path, path.name)
    return path.name
