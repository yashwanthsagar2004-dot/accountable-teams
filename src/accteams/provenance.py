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


def _dirty():
    """True or False from git; None when git is unavailable, so 'clean' is never claimed without checking."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                             text=True, timeout=10)
    except Exception:
        return None
    return None if out.returncode != 0 else bool(out.stdout.strip())


def fingerprint(config: dict) -> dict:
    """What must match for partial results to be combined: the code (commit, dirty flag) and every setting except
    the output path."""
    cfg = {k: v for k, v in config.items() if k != "out"}
    return dict(commit=_git("rev-parse", "--short", "HEAD"), dirty=_dirty(),
                config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12])


def check_resume(stored: dict | None, config: dict, what: str) -> None:
    """Refuse to resume from partial results made by other code or settings (adversarial review R1, R2, C4).
    A dirty tree never resumes, because its code cannot be identified."""
    now = fingerprint(config)
    if stored is None:
        raise RuntimeError(f"{what} has no fingerprint (written before 19 Nov 2026); use a new --out")
    if stored != now or now["dirty"] is not False:
        raise RuntimeError(f"{what} was made by {stored}, this run is {now}; refusing to mix them. "
                           "Commit your changes or use a new --out.")


def write_json_atomic(obj, path: str, **kw) -> None:
    """Write to a temporary file and rename it, so an interrupted write never leaves a truncated record."""
    import os
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, **kw)
    os.replace(path + ".tmp", path)


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
        dirty=_dirty(),
        command=" ".join(sys.argv),
        config_hash=hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12],
        python=platform.python_version(), platform=platform.platform(), device=device, versions=versions,
        seconds=round(time.time() - _T0, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        preregistered=False,
    )
