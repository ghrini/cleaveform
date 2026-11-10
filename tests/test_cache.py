"""The cross-cutting cache (spec section 7).

Two properties matter more than speed:

1. A warm run must produce byte-identical output to a cold one. A cache that
   changes the answer is worse than no cache, and this one nearly did: PLY and
   STL both round-trip vertices through float32.
2. Changing a parameter must invalidate its own stage and everything
   downstream, and nothing upstream. Too narrow an invalidation serves a stale
   result, which is a correctness bug wearing a performance costume.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cleaveform.cache import (
    STAGE_CHUNK,
    STAGE_FIELDS,
    STAGE_JOINT,
    STAGE_MESH,
    STAGE_VALIDATION,
    UNCACHED_FIELDS,
    Cache,
    CacheError,
    all_config_field_names,
    hash_mesh,
    pack,
    unpack,
)
from cleaveform.config import Config
from cleaveform.pipeline import run


def _cfg(tmp_path: Path, **overrides) -> Config:
    return Config.load(
        overrides={"cache_dir": str(tmp_path / "cache"), **overrides}
    )


def _digests(out_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((out_dir / "parts").glob("*.stl"))
    }


class TestFieldCoverage:
    """Every config field must be consciously assigned to a stage."""

    def test_no_config_field_is_unaccounted_for(self):
        covered = set().union(*STAGE_FIELDS.values()) | set(UNCACHED_FIELDS)
        missing = all_config_field_names() - covered
        assert not missing, (
            "these config fields are in neither STAGE_FIELDS nor UNCACHED_FIELDS, "
            f"so nobody decided whether they invalidate a cache: {sorted(missing)}"
        )

    def test_no_stage_lists_a_field_that_does_not_exist(self):
        covered = set().union(*STAGE_FIELDS.values()) | set(UNCACHED_FIELDS)
        assert not covered - all_config_field_names()

    def test_joint_params_are_not_in_the_chunk_stage(self):
        """Changing the clearance must not force a re-cut."""
        assert "joint_clearance_mm" not in STAGE_FIELDS[STAGE_CHUNK]
        assert "peg_radius_mm" not in STAGE_FIELDS[STAGE_CHUNK]

    def test_cut_params_are_in_the_chunk_stage(self):
        for name in ("cut_fractions", "max_parts", "design_ceiling_mm", "min_part_mm"):
            assert name in STAGE_FIELDS[STAGE_CHUNK]


class TestSerialization:
    def test_mesh_round_trip_is_exact(self):
        """float32 formats lose ~1e-6 mm, which would break determinism."""
        mesh = trimesh.creation.cylinder(radius=37.3, height=101.7, sections=64)
        _, restored = unpack(pack({}, {"m": mesh}))
        assert np.array_equal(restored["m"].vertices, mesh.vertices)
        assert np.array_equal(restored["m"].faces, mesh.faces)
        assert hash_mesh(restored["m"]) == hash_mesh(mesh)

    def test_exported_stl_bytes_survive_a_round_trip(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=11.1)
        _, restored = unpack(pack({}, {"m": mesh}))
        assert trimesh.exchange.stl.export_stl(
            restored["m"]
        ) == trimesh.exchange.stl.export_stl(mesh)

    def test_payload_survives_a_round_trip(self):
        payload, _ = unpack(pack({"a": 1, "b": ["x", None], "c": 2.5}, {}))
        assert payload == {"a": 1, "b": ["x", None], "c": 2.5}

    def test_mesh_names_with_dots_and_underscores_are_safe(self):
        """Node paths look like root.0.1 and part ids like part_001."""
        meshes = {
            "root.0.1": trimesh.creation.box(extents=(2.0, 3.0, 4.0)),
            "part_001": trimesh.creation.box(extents=(5.0, 6.0, 7.0)),
        }
        _, restored = unpack(pack({}, meshes))
        assert sorted(restored) == ["part_001", "root.0.1"]
        for name, mesh in meshes.items():
            assert hash_mesh(restored[name]) == hash_mesh(mesh)

    def test_a_corrupt_entry_is_reported_not_silently_wrong(self):
        with pytest.raises(CacheError, match="unreadable"):
            unpack(b"this is not an npz archive")


class TestStore:
    def test_a_miss_then_a_hit(self, tmp_path):
        cache = Cache(_cfg(tmp_path))
        assert cache.get(STAGE_MESH, "abc") is None
        cache.put(STAGE_MESH, "abc", b"payload")
        assert cache.get(STAGE_MESH, "abc") == b"payload"
        assert cache.stats.hits[STAGE_MESH] == 1
        assert cache.stats.misses[STAGE_MESH] == 1

    def test_a_disabled_cache_never_stores_or_serves(self, tmp_path):
        cache = Cache(_cfg(tmp_path, cache_enabled=False))
        cache.put(STAGE_MESH, "abc", b"payload")
        assert cache.get(STAGE_MESH, "abc") is None
        assert not cache.stats.hits
        assert not (tmp_path / "cache").exists()

    def test_keys_differ_when_a_stage_field_changes(self, tmp_path):
        cache = Cache(_cfg(tmp_path))
        base = _cfg(tmp_path)
        changed = base.with_overrides(min_part_mm=7.5)
        assert cache.stage_key(STAGE_CHUNK, base) != cache.stage_key(
            STAGE_CHUNK, changed
        )

    def test_keys_match_when_an_unrelated_field_changes(self, tmp_path):
        cache = Cache(_cfg(tmp_path))
        base = _cfg(tmp_path)
        changed = base.with_overrides(joint_clearance_mm=0.25)
        assert cache.stage_key(STAGE_CHUNK, base) == cache.stage_key(
            STAGE_CHUNK, changed
        )

    def test_upstream_key_propagates_downstream(self, tmp_path):
        cache = Cache(_cfg(tmp_path))
        config = _cfg(tmp_path)
        assert cache.stage_key(STAGE_JOINT, config, "upstream-a") != (
            cache.stage_key(STAGE_JOINT, config, "upstream-b")
        )

    def test_an_unknown_stage_is_refused(self, tmp_path):
        with pytest.raises(CacheError, match="unknown cache stage"):
            Cache(_cfg(tmp_path)).stage_key("not_a_stage", _cfg(tmp_path))

    def test_clear_removes_everything(self, tmp_path):
        cache = Cache(_cfg(tmp_path))
        cache.put(STAGE_MESH, "abc", b"payload")
        cache.clear()
        assert cache.get(STAGE_MESH, "abc") is None


class TestWarmRunMatchesCold:
    """The property that makes the cache safe to leave on by default."""

    def test_output_is_byte_identical_cold_and_warm(self, oversize_bar, tmp_path):
        config = _cfg(tmp_path)
        cold = run(oversize_bar, tmp_path / "cold", config)
        warm = run(oversize_bar, tmp_path / "warm", config)

        assert _digests(tmp_path / "cold") == _digests(tmp_path / "warm")
        assert cold.manifest["decomposition"]["cuts"] == (
            warm.manifest["decomposition"]["cuts"]
        )
        assert cold.manifest["joints"] == warm.manifest["joints"]
        assert cold.manifest["assembly"] == warm.manifest["assembly"]
        assert cold.manifest["constraints"] == warm.manifest["constraints"]

    def test_a_disabled_cache_gives_the_same_answer(self, oversize_bar, tmp_path):
        cached = run(oversize_bar, tmp_path / "on", _cfg(tmp_path))
        plain = run(
            oversize_bar, tmp_path / "off", _cfg(tmp_path, cache_enabled=False)
        )
        assert _digests(tmp_path / "on") == _digests(tmp_path / "off")
        assert cached.manifest["parts"] == plain.manifest["parts"]


class TestInvalidation:
    """Spec section 13: unchanged upstream param means no recompute."""

    def test_second_identical_run_recomputes_nothing(self, oversize_bar, tmp_path):
        config = _cfg(tmp_path)
        run(oversize_bar, tmp_path / "first", config)
        warm = run(oversize_bar, tmp_path / "second", config)

        stats = warm.cache_stats
        assert stats["miss_count"] == 0, (
            f"a repeat run should hit every stage, but missed: {stats['misses']}"
        )
        assert stats["hit_count"] > 0

    def test_changing_the_clearance_reuses_the_decomposition(
        self, oversize_bar, tmp_path
    ):
        """The payoff case: recalibrating must not re-cut the model."""
        config = _cfg(tmp_path)
        run(oversize_bar, tmp_path / "first", config)

        recalibrated = run(
            oversize_bar,
            tmp_path / "second",
            config.with_overrides(joint_clearance_mm=0.25),
        )
        stats = recalibrated.cache_stats
        assert STAGE_CHUNK not in stats["misses"], "decomposition should be reused"
        assert stats["hits"].get(STAGE_CHUNK, 0) >= 1
        assert stats["misses"].get(STAGE_JOINT, 0) >= 1, "joints must be rebuilt"

    def test_changing_a_cut_param_invalidates_chunk_and_downstream(
        self, oversize_bar, tmp_path
    ):
        config = _cfg(tmp_path)
        run(oversize_bar, tmp_path / "first", config)

        recut = run(
            oversize_bar,
            tmp_path / "second",
            config.with_overrides(cut_fractions=(0.45, 0.5, 0.55)),
        )
        stats = recut.cache_stats
        assert stats["hits"].get(STAGE_MESH, 0) >= 1, "the mesh stage is upstream"
        assert stats["misses"].get(STAGE_CHUNK, 0) >= 1
        assert stats["misses"].get(STAGE_JOINT, 0) >= 1, "downstream must follow"

    def test_changing_a_cost_weight_invalidates_the_decomposition(
        self, oversize_bar, tmp_path
    ):
        config = _cfg(tmp_path)
        run(oversize_bar, tmp_path / "first", config)
        reweighted = run(
            oversize_bar,
            tmp_path / "second",
            config.with_overrides(w_strength_penalty=5.0),
        )
        assert reweighted.cache_stats["misses"].get(STAGE_CHUNK, 0) >= 1

    def test_a_different_input_file_invalidates_everything(
        self, oversize_bar, tall_cylinder, tmp_path
    ):
        config = _cfg(tmp_path)
        run(oversize_bar, tmp_path / "bar", config)
        other = run(tall_cylinder, tmp_path / "cyl", config)
        assert other.cache_stats["misses"].get(STAGE_MESH, 0) >= 1

    def test_edited_input_bytes_invalidate_the_mesh_stage(
        self, oversize_bar, tmp_path
    ):
        """Keys are content hashes, so an edit in place is still detected."""
        config = _cfg(tmp_path)
        copy = tmp_path / "model.stl"
        copy.write_bytes(Path(oversize_bar).read_bytes())
        run(copy, tmp_path / "before", config)

        scaled = trimesh.load_mesh(str(oversize_bar))
        scaled.apply_scale(0.9)
        copy.write_bytes(trimesh.exchange.stl.export_stl(scaled))

        after = run(copy, tmp_path / "after", config)
        assert after.cache_stats["misses"].get(STAGE_MESH, 0) >= 1


class TestManifest:
    def test_cache_activity_is_reported(self, oversize_bar, tmp_path):
        result = run(oversize_bar, tmp_path / "out", _cfg(tmp_path))
        assert "cache" in result.manifest
        assert set(result.manifest["cache"]) >= {"hits", "misses", "hit_count"}
