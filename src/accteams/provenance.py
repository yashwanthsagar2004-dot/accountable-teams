"""Stamp a run record with what a reader needs to trust and repeat it: commit, exact command, configuration hash,
hardware, library versions and wall-clock time. The compute table of a paper is the sum of these stamps."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, time

_T0 = time.time()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def stamp(config: dict | None = None) -> dict:
    """Provenance block for one record. Call it when the run ends; `seconds` counts from import."""
    config = config or {}
    versions = {}
    for name in ("numpy", "jax", "optax", "jumanji"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            pass
    device = ""
    try:
        import jax
        device = str(jax.devices()[0])
    except Exception:
        pass
    return dict(
        commit=_git("rev-parse", "--short", "HEAD"),
        dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        command=" ".join(sys.argv),
        config_hash=hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12],
        python=platform.python_version(), platform=platform.platform(), device=device, versions=versions,
        seconds=round(time.time() - _T0, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        preregistered=False,
    )
