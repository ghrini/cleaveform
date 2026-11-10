"""Shared fixtures for the Cleaveform test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from cleaveform.config import Config

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Path:
    path = FIXTURE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run python tests/fixtures/generate_fixtures.py"
        )
    return path


@pytest.fixture(scope="session")
def config(tmp_path_factory) -> Config:
    """Shared config with the cache pointed at a temp dir.

    Sharing one cache across the suite is deliberate: if a key were wrong, a
    result computed for one test would leak into another and something would
    fail loudly rather than quietly passing.
    """
    return Config.load(
        overrides={"cache_dir": str(tmp_path_factory.mktemp("cleaveform-cache"))}
    )


@pytest.fixture(scope="session")
def small_cube() -> Path:
    """Fits the design ceiling; must come back as a single part."""
    return fixture("small_cube.stl")


@pytest.fixture(scope="session")
def oversize_bar() -> Path:
    """620 mm long, well past the bed on one axis."""
    return fixture("oversize_bar.stl")


@pytest.fixture(scope="session")
def oversize_block() -> Path:
    """Past the bed on two axes; exercises multi-axis assembly ordering."""
    return fixture("oversize_block.stl")


@pytest.fixture(scope="session")
def tall_cylinder() -> Path:
    """Round cross-sections, so joints land on non-rectangular faces."""
    return fixture("tall_cylinder.stl")


@pytest.fixture(scope="session")
def narrow_plate() -> Path:
    """Thick enough to cut, too thin to host a peg: trips a hard constraint."""
    return fixture("narrow_plate.stl")


@pytest.fixture(scope="session")
def implausibly_tiny() -> Path:
    """0.05 mm across; must be rejected without a unit override."""
    return fixture("implausibly_tiny.stl")


@pytest.fixture(scope="session")
def bar_run(oversize_bar: Path, config: Config, tmp_path_factory):
    """One shared successful run, reused by the read-only assertions."""
    from cleaveform.pipeline import run

    out = tmp_path_factory.mktemp("bar_run")
    return run(oversize_bar, out, config)
