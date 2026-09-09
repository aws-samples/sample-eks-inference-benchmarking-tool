"""General-purpose utilities with no knowledge of vLLM, HTTP or benchmarking, split from client.py by
dependency direction rather than size: anything here could be lifted into an unrelated project
unchanged. Statistics live here because three harnesses once computed percentiles three slightly
different ways, which made p99 figures from different files quietly incomparable."""
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------- #
#                                  STATISTICS                                  #
# ---------------------------------------------------------------------------- #

def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, no interpolation. q in [0, 1]; 0.0 for an empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def median(values: list[float]) -> float:
    """Middle value, 0 for an empty input, so a missing measurement never raises."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def spread_fraction(values: list[float]) -> float:
    """Peak-to-peak range as a fraction of the median: the run-to-run variance signal."""
    if len(values) < 2:
        return 0.0
    mid = median(values)
    return (max(values) - min(values)) / mid if mid > 0 else 0.0


def minmax_median(values: list[int]) -> tuple[int, int, int]:
    """Return (median, min, max) in one pass over a list, zeros when it is empty."""
    if not values:
        return 0, 0, 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2], ordered[0], ordered[-1]


# ---------------------------------------------------------------------------- #
#                                TEXT & HASHING                                #
# ---------------------------------------------------------------------------- #

def strip_trailing_whitespace(text: str) -> str:
    """Normalise line endings for hashing: drop per-line trailing space, then strip.

    Deliberately conservative — casing and punctuation are preserved, because a config
    change that alters them is exactly the kind of drift worth catching.
    """
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def sha256_text(text: str) -> str:
    """Hex digest of normalised text."""
    return hashlib.sha256(strip_trailing_whitespace(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex digest of a file's raw bytes, used to pin a dataset to a stored reference."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distinct_ngram_ratio(text: str, n: int) -> float:
    """Fraction of word n-grams that are unique; low values mean output collapsed to repetition."""
    tokens = text.split()
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------- #
#                                     TIME                                     #
# ---------------------------------------------------------------------------- #

def utc_stamp(fmt: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    """Current UTC time. Every recorded timestamp uses this, so all of them are comparable."""
    return format(datetime.now(timezone.utc), fmt)


# ---------------------------------------------------------------------------- #
#                                  SUBPROCESS                                  #
# ---------------------------------------------------------------------------- #

def shell_result(cmd: list[str], cwd: Path | None = None,
                 timeout: int = 15) -> tuple[str, str]:
    """Run a command and return (stdout, error text), never raising.

    The single try-except in the suite's subprocess path. Returning stderr as well as stdout
    is what lets a caller distinguish "nothing to report" from "the command failed" — an
    expired kubectl credential and an empty namespace both yield empty stdout, and reporting
    them identically sends the reader looking in the wrong place.
    """
    try:
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if done.returncode != 0:
        return done.stdout, (done.stderr.strip() or f"exit {done.returncode}")
    return done.stdout, ""


def shell_output(cmd: list[str], cwd: Path | None = None, timeout: int = 15) -> str:
    """Run a command and return its stdout, or "" if it fails for any reason.

    Callers that want a revision or a log excerpt should degrade to a placeholder rather than
    abort. Use shell_result() where the failure reason matters.
    """
    return shell_result(cmd, cwd=cwd, timeout=timeout)[0]


def git_revision(repo: Path, dirty_check_paths: list[str] | None = None) -> str:
    """Short HEAD revision, suffixed -dirty when the given paths have uncommitted changes.

    Returns "unknown" outside a git checkout rather than raising, because an in-cluster pod
    legitimately has no .git and the caller records the value instead of depending on it.
    """
    rev = shell_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo).strip() or "unknown"
    if rev == "unknown" or not dirty_check_paths:
        return rev
    changed = shell_output(["git", "status", "--porcelain", *dirty_check_paths], cwd=repo).strip()
    return rev + ("-dirty" if changed else "")
