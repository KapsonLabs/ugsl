"""Semantic version validation and operator-selected increments."""

import re

CORE = r"(0|[1-9][0-9]*)"
IDENTIFIERS = r"[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*"
SEMVER = re.compile(rf"{CORE}\.{CORE}\.{CORE}(?:-({IDENTIFIERS}))?(?:\+({IDENTIFIERS}))?")


def version_key(value: str) -> tuple:
    match = SEMVER.fullmatch(value)
    if not match or len(value) > 80:
        raise ValueError("Use a semantic version such as 1.0.0 or 1.0.0-rc.1.")
    major, minor, patch, prerelease, _ = match.groups()
    identifiers = prerelease.split(".") if prerelease else []
    if any(item.isdigit() and len(item) > 1 and item.startswith("0") for item in identifiers):
        raise ValueError("Numeric prerelease identifiers cannot have leading zeroes.")
    ranking = tuple((0, int(item)) if item.isdigit() else (1, item) for item in identifiers)
    return int(major), int(minor), int(patch), not prerelease, ranking, value


def next_version(versions: list[str], bump: str) -> str:
    if bump not in {"patch", "minor", "major"}:
        raise ValueError("Version increment must be patch, minor or major.")
    if not versions:
        return "1.0.0"
    major, minor, patch, *_ = max(map(version_key, versions))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"
