"""Command line entry point.

    cleaveform model.stl -o out/     split a model into printable parts
    cleaveform calibrate -o calib/   generate the joint-fit calibration prints
    cleaveform calibrate --mode brick -o fit/   the brick clutch fixture
    cleaveform config                print the active configuration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import __version__
from .calibration import (
    BRICK_README_FILENAME,
    GAUGE_README_FILENAME,
    write_brick_calibration_set,
    write_brick_gauge_set,
    write_calibration_set,
)
from .config import Config
from .cache import Cache
from .estimate import MINUTES_PER_HOUR
from .export import DIAGNOSTIC_FILENAME
from .geometry import UnitError
from .io import InputError, parse_target_size
from .pipeline import PipelineError, run

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _load_config(config_path: str | None, overrides: dict) -> Config:
    return Config.load(config_path, overrides)


def _echo(message: str) -> None:
    click.echo(f"  {message}")


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.version_option(__version__, prog_name="cleaveform")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Split a 3D model into printable chunks that cleave back together."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_dir", required=True, type=click.Path(file_okay=False),
              help="Output directory.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              help="JSON file of config overrides.")
@click.option("--input-units", type=click.Choice(["mm", "cm", "m", "in"]),
              help="Interpret input coordinates in these units.")
@click.option("--target-size", help="Scale to fit X,Y,Z millimetres.")
@click.option("--scale-factor", type=float, help="Multiply input coordinates by this.")
@click.option("--joint-clearance", type=float,
              help="Calibrated joint clearance in mm (from cleaveform calibrate).")
@click.option("--max-parts", type=int, help="Override the part-count ceiling.")
@click.option("--max-cut-depth", type=int, help="Override the recursion depth ceiling.")
@click.option("--mode", "decomposition_mode", type=click.Choice(["chunk", "brick"]),
              help="chunk: cut into a few large custom pieces (default). "
                   "brick: build from small standard interlocking bricks.")
@click.option("--brick-unit", type=float,
              help="Stud pitch in mm for brick mode (default 8).")
@click.option("--simplify", is_flag=True,
              help="Brick mode: change the shape minimally so every brick can be held "
                   "(trim slivers, thicken flat floors, stitch seams). What changed is reported.")
@click.option("--keep-thin", is_flag=True,
              help="Brick mode: keep features thinner than a brick (rails, bridges) where "
                   "they join separate parts.")
@click.option("--no-cache", is_flag=True,
              help="Recompute everything; do not read or write the cache.")
@click.option("--clear-cache", is_flag=True,
              help="Delete the cache before running.")
@click.option("--cache-dir", type=click.Path(file_okay=False),
              help="Where cached stage results live.")
@click.option("--experimental", is_flag=True,
              help="Allow recovery to explore clearance. Output is not production safe.")
@click.option("-q", "--quiet", is_flag=True, help="Only report the final result.")
def split(
    input_file: str,
    out_dir: str,
    config_path: str | None,
    input_units: str | None,
    target_size: str | None,
    scale_factor: float | None,
    joint_clearance: float | None,
    max_parts: int | None,
    max_cut_depth: int | None,
    decomposition_mode: str | None,
    brick_unit: float | None,
    simplify: bool,
    keep_thin: bool,
    no_cache: bool,
    clear_cache: bool,
    cache_dir: str | None,
    experimental: bool,
    quiet: bool,
) -> None:
    """Split INPUT_FILE into printable parts, as chunks or as bricks."""
    _run_split(
        input_file, out_dir, config_path, input_units, target_size, scale_factor,
        joint_clearance, max_parts, max_cut_depth, decomposition_mode, brick_unit,
        simplify, keep_thin, no_cache, clear_cache, cache_dir, experimental, quiet,
    )


def _run_split(
    input_file: str,
    out_dir: str,
    config_path: str | None,
    input_units: str | None,
    target_size: str | None,
    scale_factor: float | None,
    joint_clearance: float | None,
    max_parts: int | None,
    max_cut_depth: int | None,
    decomposition_mode: str | None,
    brick_unit: float | None,
    simplify: bool,
    keep_thin: bool,
    no_cache: bool,
    clear_cache: bool,
    cache_dir: str | None,
    experimental: bool,
    quiet: bool,
) -> None:
    overrides = {
        "joint_clearance_mm": joint_clearance,
        "max_parts": max_parts,
        "max_cut_depth": max_cut_depth,
        "decomposition_mode": decomposition_mode,
        "brick_unit_mm": brick_unit,
        "brick_simplify": True if simplify else None,
        "brick_keep_thin": True if keep_thin else None,
        "cache_enabled": False if no_cache else None,
        "cache_dir": cache_dir,
        "experimental": True if experimental else None,
    }
    try:
        config = _load_config(config_path, overrides)
        target = parse_target_size(target_size)
    except (ValueError, InputError) as exc:
        raise click.ClickException(str(exc)) from exc

    if clear_cache:
        Cache(config).clear()
        if not quiet:
            click.echo(f"cleared cache at {config.cache_dir}")

    if experimental:
        click.secho(
            "experimental run: joint clearance may deviate from the calibrated value. "
            "Do not use these parts as a production reference.",
            fg="yellow",
        )

    log = (lambda message: None) if quiet else _echo
    if not quiet:
        click.echo(f"cleaveform {__version__}")

    try:
        result = run(
            input_file,
            out_dir,
            config,
            input_units=input_units,
            target_size_mm=target,
            scale_factor=scale_factor,
            on_progress=log,
        )
    except (InputError, UnitError) as exc:
        raise click.ClickException(str(exc)) from exc
    except PipelineError as exc:
        click.secho(f"\nrun failed: {exc}", fg="red", err=True)
        click.echo(
            f"see {Path(out_dir) / DIAGNOSTIC_FILENAME} for the offending part and "
            "constraint.",
            err=True,
        )
        sys.exit(1)

    totals = (result.manifest or {}).get("totals", {})
    brick_mode = (result.manifest or {}).get("mode") == "brick"
    click.echo("")
    if brick_mode:
        click.secho(
            f"done: {totals.get('brick_count', 0)} brick(s) from "
            f"{totals.get('brick_type_count', 0)} shape(s)",
            fg="green",
        )
        pieces = totals.get("piece_count", 1)
        if pieces <= 1:
            click.secho("  builds as 1 piece", fg="green")
        else:
            click.secho(
                f"  builds as {pieces} pieces (largest stray group: "
                f"{totals.get('largest_stray_bricks', 0)} brick(s))",
                fg="yellow",
            )
            if not (config.brick_simplify and config.brick_keep_thin):
                click.echo("  try --simplify (and --keep-thin for rails or thin necks) to attach them")
        click.echo(f"  shapes   {Path(out_dir) / 'bricks'}")
        click.echo(
            f"  plates   {Path(out_dir) / 'plates'} "
            f"({totals.get('plate_count', 0)} to print)"
        )
    else:
        click.secho(f"done: {totals.get('part_count', 0)} part(s) written", fg="green")
        click.echo(f"  parts    {Path(out_dir) / 'parts'}")
    click.echo(f"  metadata {result.metadata_path}")
    click.echo(f"  guide    {result.guide_path}")
    click.echo(
        f"  estimates ~{totals.get('est_total_print_time_min', 0) / MINUTES_PER_HOUR:.1f} h print, "
        f"~{totals.get('est_total_weight_g', 0):.0f} g filament (not slicer output)"
    )
    stats = result.cache_stats or {}
    if not quiet and stats.get("hit_count"):
        click.echo(
            f"  cache     {stats['hit_count']} stage(s) reused, "
            f"{stats.get('miss_count', 0)} recomputed"
        )

    if result.warnings:
        click.echo("")
        click.secho("advisories:", fg="yellow")
        for warning in result.warnings:
            click.echo(f"  - {warning}")


@main.command()
@click.option("-o", "--out", "out_dir", required=True, type=click.Path(file_okay=False),
              help="Output directory.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              help="JSON file of config overrides.")
@click.option("--mode", "mode", type=click.Choice(["chunk", "brick", "gauge"]), default="chunk",
              show_default=True,
              help="Which fit to calibrate: chunk peg joints, or the brick clutch.")
def calibrate(out_dir: str, config_path: str | None, mode: str) -> None:
    """Generate the physical fit-calibration prints.

    The two modes calibrate genuinely different things and are not
    interchangeable. A chunk joint engages about 6 mm deep and is set by a
    clearance; a brick clutch engages about 2 mm and is set by an interference,
    which is the opposite sign. A value carried across from one to the other is
    not conservative, it is simply wrong.
    """
    try:
        config = _load_config(config_path, {})
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if mode == "gauge":
        written = write_brick_gauge_set(out_dir, config)
        readme, field = GAUGE_README_FILENAME, "brick_fit_bias_mm"
    elif mode == "brick":
        written = write_brick_calibration_set(out_dir, config)
        readme, field = BRICK_README_FILENAME, "brick_clutch_interference_mm"
    else:
        written = write_calibration_set(out_dir, config)
        readme, field = "CALIBRATION.txt", "joint_clearance_mm"

    stls = [p for p in written if p.suffix == ".stl"]
    click.secho(f"wrote {len(stls)} calibration STL(s) to {out_dir}", fg="green")
    click.echo(
        f"print them all in one job, then read {readme} to pick {field}."
    )


@main.command(name="bricktest")
@click.option("-o", "--out", "out_dir", required=True, type=click.Path(file_okay=False),
              help="Output directory for the scoreboard and pictures.")
@click.option("--sizes", default=None,
              help="Comma-separated brick sizes in mm. Default: 6,8,12,16.")
@click.option("--size-mm", "size_mm", default=None, type=float,
              help="Size of every test shape's largest dimension in mm. Default: 140.")
@click.option("--only", default=None, help="Comma-separated shape names to run.")
@click.option("--category", default=None, help="Comma-separated categories to run.")
@click.option("--no-images", is_flag=True, help="Skip the side-by-side pictures.")
@click.option("--simplify", is_flag=True, help="Score with shape simplification on.")
@click.option("--keep-thin", is_flag=True, help="Score with thin-feature preservation on.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              help="JSON file of config overrides.")
def bricktest(out_dir: str, sizes: str | None, size_mm: float | None, only: str | None,
              category: str | None, no_images: bool, simplify: bool, keep_thin: bool,
              config_path: str | None) -> None:
    """Score brick mode on a catalogue of test shapes and write a scoreboard.

    Needs no printer and no model files: the shapes are generated. Each shape is
    run at several brick sizes and scored on whether it builds as one piece, how
    closely it matches the original, whether it stands up, and what it costs.
    """
    from .brick import benchmark, shapes

    try:
        config = _load_config(
            config_path,
            {"brick_simplify": True if simplify else None,
             "brick_keep_thin": True if keep_thin else None},
        )
        chosen = shapes.select(
            [n.strip() for n in only.split(",")] if only else None,
            [c.strip() for c in category.split(",")] if category else None,
        )
        unit_sizes = (
            tuple(float(v) for v in sizes.split(",")) if sizes else benchmark.DEFAULT_SIZES_MM
        )
    except (ValueError, KeyError) as exc:
        raise click.ClickException(str(exc).strip("'\"")) from exc
    if not chosen:
        raise click.ClickException("no shapes match that selection")

    size = size_mm or shapes.DEFAULT_SIZE_MM
    click.echo(f"scoring {len(chosen)} shape(s) at {', '.join(f'{u:g}' for u in unit_sizes)} mm bricks, "
               f"each {size:g} mm across")
    rows, kept = benchmark.run(chosen, unit_sizes, size, config, progress=click.echo)

    images: dict[str, str] = {}
    if not no_images:
        directory = Path(out_dir) / "images"
        made = 0
        for name, (mesh, grid, layout) in kept.items():
            if benchmark.render_pair(mesh, grid, layout, directory / f"{name}.png"):
                images[name] = f"images/{name}.png"
                made += 1
        if not made and kept:
            click.secho("matplotlib is not installed, so no pictures were drawn "
                        "(pip install matplotlib)", fg="yellow")

    written = benchmark.write_scoreboard(rows, out_dir, images)
    click.echo("")
    counts = benchmark.summary_counts(rows)
    for unit in sorted(counts):
        c = counts[unit]
        click.echo(f"  {unit:>4g} mm   pass {c['pass']:>2}   split {c['split']:>2}   "
                   f"fail {c['fail']:>2}   error {c['error']:>2}")
    click.secho(f"\nscoreboard: {written['html']}", fg="green")


@main.command(name="config")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              help="JSON file of config overrides to merge in.")
def show_config(config_path: str | None) -> None:
    """Print the active configuration as JSON."""
    try:
        config = _load_config(config_path, {})
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(config.to_dict(), indent=2))


# `cleaveform model.stl -o out/` is the documented Phase 1 invocation, so a bare
# file argument routes to `split` without the user naming the subcommand.
_ORIGINAL_MAIN = main


def entrypoint() -> None:
    argv = sys.argv[1:]
    known = {"split", "calibrate", "config", "bricktest"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        sys.argv.insert(1, "split")
    _ORIGINAL_MAIN()


if __name__ == "__main__":
    entrypoint()
