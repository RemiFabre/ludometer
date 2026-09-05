"""The smallest file store the loop needs, with a Hub backend and a local one.

Four operations: put a file, get a file, list names under a prefix, delete.
:class:`LocalHub` is a directory (tests, dry runs); :class:`HfHub` is one
Hugging Face repo, every call retried with backoff because a fleet of jobs
committing shards to one dataset repo *will* hit rate limits and concurrent-
commit conflicts now and then.

On top of that, :func:`publish_weights` / :func:`fetch_weights` implement the
one protocol both sides share: weights live at ``<run>/weights-v<N>.pt`` and a
tiny ``<run>/current.json`` names the newest ``N``. Because a file is never
rewritten in place, a reader that saw version ``N`` in the pointer downloads
exactly version ``N``, whatever is being uploaded meanwhile.

``hub_from_spec`` turns a config string into a backend: an existing directory
(or ``file:...``) is local, anything else is ``[type:]owner/name`` on the Hub
(``dataset:`` unless said otherwise).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = [
    "HfHub",
    "Hub",
    "LocalHub",
    "current_version",
    "fetch_weights",
    "hub_from_spec",
    "publish_weights",
]


class Hub(Protocol):
    def put(self, local: str | os.PathLike[str], remote: str) -> None: ...
    def put_bytes(self, data: bytes, remote: str) -> None: ...
    def get(self, remote: str, local: str | os.PathLike[str]) -> Path: ...
    def get_bytes(self, remote: str) -> bytes | None: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, remote: str) -> None: ...
    def describe(self) -> str: ...


# ----------------------------------------------------------------- local dir
class LocalHub:
    """A directory. Same semantics, no network — what the tests drive."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, remote: str) -> Path:
        p = (self.root / remote).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"remote path escapes the hub root: {remote}")
        return p

    def put(self, local: str | os.PathLike[str], remote: str) -> None:
        target = self._path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".uploading")
        shutil.copyfile(local, tmp)
        os.replace(tmp, target)

    def put_bytes(self, data: bytes, remote: str) -> None:
        target = self._path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".uploading")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    def get(self, remote: str, local: str | os.PathLike[str]) -> Path:
        src = self._path(remote)
        dst = Path(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def get_bytes(self, remote: str) -> bytes | None:
        p = self._path(remote)
        return p.read_bytes() if p.exists() else None

    def list(self, prefix: str = "") -> list[str]:
        out = []
        for p in self.root.rglob("*"):
            if p.is_file() and not p.name.endswith(".uploading"):
                rel = p.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def delete(self, remote: str) -> None:
        p = self._path(remote)
        if p.exists():
            p.unlink()

    def describe(self) -> str:
        return f"local:{self.root}"


# ----------------------------------------------------------------- hub repo
def _retry(fn: Any, tries: int = 6, base: float = 2.0, what: str = "") -> Any:
    last: BaseException | None = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - the whole point is to retry
            last = exc
            if attempt == tries - 1:
                break
            delay = base * (2**attempt) + np.random.default_rng().uniform(0, 1)
            print(
                f"[hub] {what or 'call'} failed ({type(exc).__name__}: {exc}); retry in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
    assert last is not None
    raise last


class HfHub:
    """One Hugging Face repo (``dataset`` by default), with retries."""

    def __init__(
        self, repo_id: str, repo_type: str = "dataset", token: str | None = None
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(token=token)
        self._ensure()

    def _ensure(self) -> None:
        _retry(
            lambda: self.api.create_repo(
                self.repo_id, repo_type=self.repo_type, private=True, exist_ok=True
            ),
            what=f"create {self.repo_id}",
        )

    def put(self, local: str | os.PathLike[str], remote: str) -> None:
        _retry(
            lambda: self.api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"put {remote}",
            ),
            what=f"put {remote}",
        )

    def put_bytes(self, data: bytes, remote: str) -> None:
        _retry(
            lambda: self.api.upload_file(
                path_or_fileobj=io.BytesIO(data),
                path_in_repo=remote,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"put {remote}",
            ),
            what=f"put {remote}",
        )

    def get(self, remote: str, local: str | os.PathLike[str]) -> Path:
        dst = Path(local)
        dst.parent.mkdir(parents=True, exist_ok=True)

        def _dl() -> Path:
            with tempfile.TemporaryDirectory() as td:
                got = self.api.hf_hub_download(
                    self.repo_id,
                    remote,
                    repo_type=self.repo_type,
                    local_dir=td,
                    force_download=True,
                )
                shutil.move(got, dst)
            return dst

        return _retry(_dl, what=f"get {remote}")

    def get_bytes(self, remote: str) -> bytes | None:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        def _dl() -> bytes | None:
            try:
                with tempfile.TemporaryDirectory() as td:
                    got = self.api.hf_hub_download(
                        self.repo_id,
                        remote,
                        repo_type=self.repo_type,
                        local_dir=td,
                        force_download=True,
                    )
                    return Path(got).read_bytes()
            except (EntryNotFoundError, RepositoryNotFoundError):
                return None

        return _retry(_dl, what=f"get {remote}")

    def list(self, prefix: str = "") -> list[str]:
        files = _retry(
            lambda: self.api.list_repo_files(self.repo_id, repo_type=self.repo_type),
            what=f"list {self.repo_id}",
        )
        return sorted(f for f in files if f.startswith(prefix))

    def delete(self, remote: str) -> None:
        from huggingface_hub.errors import EntryNotFoundError

        def _rm() -> None:
            try:
                self.api.delete_file(remote, self.repo_id, repo_type=self.repo_type)
            except EntryNotFoundError:
                pass

        _retry(_rm, what=f"delete {remote}")

    def describe(self) -> str:
        return f"{self.repo_type}:{self.repo_id}"


def hub_from_spec(spec: str, token: str | None = None) -> Hub:
    """``file:/dir`` or an existing directory -> LocalHub; ``[type:]owner/name`` -> HfHub."""
    if spec.startswith("file:"):
        return LocalHub(spec[len("file:") :])
    if Path(spec).is_dir() or spec.startswith(("/", "./", "../")):
        return LocalHub(spec)
    repo_type = "dataset"
    if ":" in spec:
        repo_type, spec = spec.split(":", 1)
    return HfHub(spec, repo_type=repo_type, token=token)


# ------------------------------------------------------------ weights protocol
def _pointer(run: str) -> str:
    return f"{run}/current.json"


def _weights_name(run: str, version: int) -> str:
    return f"{run}/weights-v{version:05d}.pt"


def current_version(hub: Hub, run: str) -> dict[str, Any] | None:
    raw = hub.get_bytes(_pointer(run))
    return None if raw is None else json.loads(raw.decode("utf-8"))


def publish_weights(
    hub: Hub,
    run: str,
    weights: dict[str, np.ndarray],
    net_config: dict[str, Any],
    version: int,
    extra: dict[str, Any] | None = None,
    keep: int = 3,
) -> dict[str, Any]:
    """Upload ``weights`` as version ``version`` and move the pointer to it.

    The file goes up first, the pointer second, so nobody can read a pointer to
    a file that is not there yet. Versions older than the newest ``keep`` are
    deleted afterwards (best effort) so the repo does not grow forever.
    """
    import torch

    payload = {
        "format": "ludometer-cloud-weights-1",
        "version": int(version),
        "net_config": dict(net_config),
        "weights": {
            k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in weights.items()
        },
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "w.pt"
        torch.save(payload, path)
        hub.put(path, _weights_name(run, version))
    pointer = {
        "version": int(version),
        "file": _weights_name(run, version),
        "published": time.time(),
        **(extra or {}),
    }
    hub.put_bytes(json.dumps(pointer).encode("utf-8"), _pointer(run))
    if keep > 0:
        for name in hub.list(f"{run}/weights-v"):
            try:
                v = int(name.rsplit("-v", 1)[1].split(".")[0])
            except ValueError:
                continue
            if v <= version - keep:
                try:
                    hub.delete(name)
                except Exception as exc:  # noqa: BLE001 - housekeeping only
                    print(f"[hub] could not delete {name}: {exc}", flush=True)
    return pointer


def fetch_weights(
    hub: Hub, run: str, known_version: int, into: str | os.PathLike[str]
) -> tuple[int, dict[str, Any], dict[str, np.ndarray]] | None:
    """The newest weights if their version is above ``known_version``, else None."""
    import torch

    pointer = current_version(hub, run)
    if pointer is None or int(pointer["version"]) <= known_version:
        return None
    version = int(pointer["version"])
    local = Path(into) / f"weights-v{version:05d}.pt"
    hub.get(pointer["file"], local)
    payload = torch.load(local, map_location="cpu", weights_only=False)
    weights = {k: v.numpy() for k, v in payload["weights"].items()}
    return version, dict(payload["net_config"]), weights
