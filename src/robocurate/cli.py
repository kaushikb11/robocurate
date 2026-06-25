"""Command-line interface — the surface shape for the v1 command set.

Commands mirror the Python API: ``score`` (report only, no write), ``curate`` (select +
write a new dataset + manifest), ``baseline`` (emit the equal-N random control), ``report``
(render a scorecard from a run), and ``diff`` (raw vs curated). Each accepts ``--seed`` and
``--json`` and records enough to reproduce a run from config + seed + code version.

In this skeleton the parser and command wiring are real, but several commands are thin: the
end-to-end ``curate``/``score`` flow runs because no real signal is required (signals are
resolved from the registry and, when none are registered, the command reports that clearly
rather than fabricating a result).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robocurate import __version__, signals
from robocurate.adapters.lerobot import LeRobotReader
from robocurate.curator import Budget, Curator
from robocurate.scorecard import Scorecard

if TYPE_CHECKING:
    from robocurate.signals.base import Signal


def _resolve_signals(names: list[str]) -> list[Signal]:
    """Instantiate signals by name from the registry, erroring clearly if unknown/empty."""
    if not names:
        raise SystemExit(
            "no signals specified. Pass --signals <name,...>; registered signals: "
            f"{', '.join(signals.available()) or '<none yet — this is a skeleton>'}"
        )
    resolved: list[Signal] = []
    for name in names:
        try:
            resolved.append(signals.get(name))
        except (KeyError, ImportError) as exc:
            # KeyError: unknown signal. ImportError: a known signal whose optional ML
            # extra is not installed. Both carry an actionable message.
            raise SystemExit(str(exc).strip("\"'")) from exc
    return resolved


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _cmd_score(args: argparse.Namespace) -> int:
    reader = LeRobotReader(args.dataset)
    curator = Curator(_resolve_signals(_split_csv(args.signals)), seed=args.seed)
    result = curator.run(reader)
    card = result.scorecard()
    print(card.to_json() if args.json else card.to_markdown())
    return 0


def _cmd_curate(args: argparse.Namespace) -> int:
    reader = LeRobotReader(args.dataset)
    budget = Budget.fraction(args.budget) if args.budget is not None else None
    curator = Curator(
        _resolve_signals(_split_csv(args.signals)),
        budget=budget,
        seed=args.seed,
        emit_baseline=not args.no_baseline,
    )
    result = curator.run(reader)
    receipt = result.save(args.out)
    msg = {
        "out": str(receipt.path),
        "kept": result.num_kept,
        "removed": result.num_removed,
        "manifest": str(receipt.manifest_path),
    }
    print(json.dumps(msg) if args.json else msg)
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    reader = LeRobotReader(args.dataset)
    curator = Curator(
        _resolve_signals(_split_csv(args.signals)),
        budget=Budget.count(args.n),
        seed=args.seed,
        emit_baseline=True,
    )
    result = curator.run(reader)
    assert result.baseline is not None
    print(result.baseline.to_dict() if not args.json else result.scorecard().to_json())
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Render a scorecard from a saved curation manifest (markdown, or ``--json``)."""
    path = Path(args.path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"no manifest found at {args.path} (expected a manifest.json file)")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read manifest {path}: {exc}") from exc
    card = Scorecard.from_manifest(manifest)
    print(card.to_json() if args.json else card.to_markdown())
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Report which source episodes are absent from a curated dataset (read-only)."""
    source = LeRobotReader(args.source)
    curated = LeRobotReader(args.curated)

    curated_fingerprints = {t.meta.fingerprint for t in curated}
    curated_indices = {t.meta.episode_index for t in curated}
    # Content fingerprints are the authoritative match: a curated dataset re-indexes its
    # kept episodes, so episode index alone is unreliable. Only fall back to index matching
    # when no usable fingerprints exist on the curated side (an unfingerprinted source).
    use_fingerprints = bool(curated_fingerprints - {""})

    removed: list[dict[str, Any]] = []
    for traj in source:
        present = (
            traj.meta.fingerprint in curated_fingerprints
            if use_fingerprints
            else traj.meta.episode_index in curated_indices
        )
        if not present:
            removed.append(
                {
                    "episode_index": traj.meta.episode_index,
                    "fingerprint": traj.meta.fingerprint,
                }
            )

    n_source = len(source)
    n_curated = len(curated)
    if args.json:
        print(
            json.dumps(
                {
                    "n_source": n_source,
                    "n_curated": n_curated,
                    "n_removed": len(removed),
                    "removed": removed,
                }
            )
        )
        return 0

    lines = [
        f"source: {n_source} episodes; curated: {n_curated} episodes; removed: {len(removed)}.",
    ]
    if removed:
        lines.append("Removed episodes (absent from curated):")
        lines += [f"  - episode {r['episode_index']} ({r['fingerprint'][:12]})" for r in removed]
    else:
        lines.append("No source episodes are missing from the curated dataset.")
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser (the CLI surface shape)."""
    parser = argparse.ArgumentParser(prog="robocurate", description=__doc__)
    parser.add_argument("--version", action="version", version=f"robocurate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=0, help="master seed (determinism).")
        p.add_argument("--json", action="store_true", help="emit machine-readable JSON.")

    p_score = sub.add_parser("score", help="score a dataset and print a scorecard (no write).")
    p_score.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_score.add_argument("--signals", help="comma-separated signal names.")
    add_common(p_score)
    p_score.set_defaults(func=_cmd_score)

    p_curate = sub.add_parser("curate", help="select a subset and write a new dataset.")
    p_curate.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_curate.add_argument("--out", required=True, help="destination for the curated dataset.")
    p_curate.add_argument("--signals", help="comma-separated signal names.")
    p_curate.add_argument("--budget", type=float, help="fraction of episodes to keep (0-1].")
    p_curate.add_argument(
        "--no-baseline", action="store_true", help="skip the equal-N random baseline."
    )
    add_common(p_curate)
    p_curate.set_defaults(func=_cmd_curate)

    p_base = sub.add_parser("baseline", help="emit an equal-N random control selection.")
    p_base.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_base.add_argument("--n", type=int, required=True, help="number of episodes to keep.")
    p_base.add_argument("--signals", help="comma-separated signal names.")
    add_common(p_base)
    p_base.set_defaults(func=_cmd_baseline)

    p_report = sub.add_parser("report", help="render a scorecard from a saved run.")
    p_report.add_argument("path", help="path to a manifest or curated dataset.")
    add_common(p_report)
    p_report.set_defaults(func=_cmd_report)

    p_diff = sub.add_parser("diff", help="diff a source dataset against a curated one.")
    p_diff.add_argument("source", help="source dataset path.")
    p_diff.add_argument("curated", help="curated dataset path.")
    add_common(p_diff)
    p_diff.set_defaults(func=_cmd_diff)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = args.func
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
