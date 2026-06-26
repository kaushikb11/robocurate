"""Command-line interface — the surface shape for the v1 command set.

Commands mirror the Python API: ``score`` (report only, no write), ``curate`` (select +
write a new dataset + manifest), ``baseline`` (emit the equal-N random control), ``report``
(render a scorecard from a run), ``diff`` (raw vs curated), ``list-signals`` (the available
quality signals), ``validate``/``doctor`` (a read-only dataset health check), and ``explain``
(one episode's kept/removed decision from a saved manifest). Each accepts ``--seed`` and
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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robocurate import __version__, signals
from robocurate.adapters.lerobot import LeRobotReader
from robocurate.curator import Budget, Curator
from robocurate.scorecard import Scorecard

if TYPE_CHECKING:
    from robocurate.signals.base import Signal, SignalSpec


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


def _curate_curator(args: argparse.Namespace) -> Curator:
    """Build the curator for a ``curate`` run: from a recipe, or from --signals/--budget.

    A recipe is mutually exclusive with --signals/--budget: a recipe already fixes the full
    config (combiner, budget, selection, gate, seed), so mixing the two would be ambiguous.
    """
    from robocurate.recipe import load_recipe

    if args.recipe is not None:
        if args.signals or args.budget is not None:
            raise SystemExit(
                "--recipe is mutually exclusive with --signals/--budget: a recipe already "
                "fixes the full curation config. Pass one or the other, not both."
            )
        return load_recipe(args.recipe)
    budget = Budget.fraction(args.budget) if args.budget is not None else None
    return Curator(
        _resolve_signals(_split_csv(args.signals)),
        budget=budget,
        seed=args.seed,
        emit_baseline=not args.no_baseline,
    )


def _cmd_curate(args: argparse.Namespace) -> int:
    from robocurate.recipe import save_recipe

    reader = LeRobotReader(args.dataset)
    curator = _curate_curator(args)
    result = curator.run(reader)
    receipt = result.save(
        args.out,
        write_card=not args.no_card,
        push_to_hub=args.push_to_hub,
    )
    if args.report_html is not None:
        Path(args.report_html).write_text(result.scorecard().to_html(), encoding="utf-8")
    if args.save_recipe is not None:
        save_recipe(curator, args.save_recipe)
    msg: dict[str, Any] = {
        "out": str(receipt.path),
        "kept": result.num_kept,
        "removed": result.num_removed,
        "manifest": str(receipt.manifest_path),
    }
    if args.report_html is not None:
        msg["report_html"] = str(args.report_html)
    if args.save_recipe is not None:
        msg["recipe"] = str(args.save_recipe)
    if args.push_to_hub is not None:
        msg["pushed_to_hub"] = args.push_to_hub
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


# Built-in signals whose optional extra differs from their signal name. Most learned
# signals install via ``robocurate[<name>]``; these two share an umbrella extra. TIER0_CPU
# signals need no extra and report "(none)".
_SIGNAL_EXTRA = {
    "demo_score": "demo-score",
    "cupid": "influence",
}


def _signal_extra(spec: SignalSpec) -> str | None:
    """Return the pip extra a signal needs, or ``None`` for a laptop-friendly Tier-0 signal."""
    from robocurate.signals.base import CostTier

    if spec.cost_tier is CostTier.TIER0_CPU:
        return None
    return _SIGNAL_EXTRA.get(spec.name, spec.name)


def _signal_entry(name: str) -> dict[str, Any] | None:
    """Build the descriptor for one available signal (name, spec fields, required extra).

    Some learned signals import their class successfully (so they appear in ``available()``)
    but raise at construction when an optional dependency (e.g. PyTorch) is missing. Those are
    reported with the install hint we can derive from the name rather than dropped or crashed.
    """
    try:
        spec = signals.get(name).spec
    except ImportError as exc:
        return {
            "name": name,
            "description": f"needs an optional dependency to load ({exc})",
            "cost_tier": None,
            "requires": [],
            "deterministic": None,
            "extra": _SIGNAL_EXTRA.get(name, name),
        }
    return {
        "name": spec.name,
        "description": spec.description,
        "cost_tier": spec.cost_tier.name,
        "requires": sorted(spec.requires),
        "deterministic": spec.deterministic,
        "extra": _signal_extra(spec),
    }


def _cmd_list_signals(args: argparse.Namespace) -> int:
    """List every loadable signal (plus any that failed to import) — JSON or markdown."""
    entries = (_signal_entry(name) for name in signals.available())
    available = [entry for entry in entries if entry is not None]
    unavailable = signals.unavailable()

    if args.json:
        print(json.dumps({"available": available, "unavailable": unavailable}))
        return 0

    lines = ["# Available signals", ""]
    if not available:
        lines.append("(no signals registered)")
    for entry in available:
        extra = entry["extra"] or "none (Tier-0, CPU)"
        requires = ", ".join(entry["requires"]) or "—"
        cost_tier = entry["cost_tier"] or "unknown (dependency missing)"
        if entry["deterministic"] is None:
            deterministic = "unknown"
        else:
            deterministic = "yes" if entry["deterministic"] else "no"
        lines += [
            f"## {entry['name']}",
            "",
            f"  {entry['description']}",
            "",
            f"  - cost tier:     {cost_tier}",
            f"  - requires:      {requires}",
            f"  - deterministic: {deterministic}",
            f"  - install extra: {extra}",
            "",
        ]
    if unavailable:
        lines += ["## Unavailable (import failed)", ""]
        for name, error in sorted(unavailable.items()):
            lines.append(f"  - {name}: {error}")
    print("\n".join(lines).rstrip())
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Diagnose a source dataset's health (schema, structural defects, coverage). No write."""
    from robocurate.dataset import Dataset
    from robocurate.health import dataset_health

    dataset = Dataset.from_lerobot(args.dataset)
    report = dataset_health(dataset)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_markdown())
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """Explain one episode's kept/removed decision from a saved manifest (markdown or JSON)."""
    path = Path(args.path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"no manifest found at {args.path} (expected a manifest.json file)")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read manifest {path}: {exc}") from exc

    decisions = manifest.get("decisions", [])
    match = next((d for d in decisions if d.get("episode_index") == args.episode_index), None)
    if match is None:
        known = ", ".join(str(d.get("episode_index")) for d in decisions) or "<none>"
        raise SystemExit(
            f"episode {args.episode_index} not found in {path}; episodes present: {known}"
        )

    if args.json:
        print(json.dumps(match))
        return 0

    status = "KEPT" if match.get("kept") else "REMOVED"
    fingerprint = str(match.get("fingerprint", ""))
    lines = [
        f"# Episode {match.get('episode_index')}: {status}",
        "",
        f"- fingerprint: {fingerprint[:12] or '(none)'}",
        f"- reason: {match.get('reason', '(no reason recorded)')}",
        "",
        "## Per-signal values",
        "",
    ]
    signal_values = match.get("signal_values") or {}
    if signal_values:
        for name, value in signal_values.items():
            lines.append(f"  - {name}: {value}")
    else:
        lines.append("  (no per-signal values recorded)")
    print("\n".join(lines))
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


def _open_pool_reader(path: str) -> Any:
    """Open a LeRobotDataset directory as a read-only reader (v2.1 or v3, auto-detected)."""
    from robocurate.dataset import Dataset

    return Dataset.from_lerobot(path).reader


def _cmd_benchmark_init(args: argparse.Namespace) -> int:
    """Build a frozen benchmark spec from a pool dataset and write it to ``--out``."""
    from robocurate.benchmark.spec import build_spec

    reader = _open_pool_reader(args.pool)
    spec = build_spec(reader, eval_frac=args.eval_frac, seed=args.seed)
    Path(args.out).write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    msg = {
        "out": str(args.out),
        "pool": spec.pool.dataset_id,
        "n_eval": len(spec.eval_split_indices),
        "n_train_pool": len(spec.train_pool_indices),
    }
    print(json.dumps(msg) if args.json else msg)
    return 0


def _cmd_benchmark_run(args: argparse.Namespace) -> int:
    """Score a submission against a spec; print the result and optionally append to a board."""
    from robocurate.benchmark.leaderboard import append_entry
    from robocurate.benchmark.runner import run_submission
    from robocurate.benchmark.spec import BenchmarkSpec

    spec = BenchmarkSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    reader = _open_pool_reader(args.pool) if args.pool else _open_pool_reader(spec.pool.dataset_id)
    result = run_submission(spec, args.submission, reader, master_seed=args.seed)

    name = args.name if args.name is not None else _default_name(args.submission)
    if args.leaderboard is not None:
        append_entry(args.leaderboard, result, name=name, created_utc=args.created_utc)

    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print(_result_markdown(result, name=name))
    return 0


def _cmd_benchmark_leaderboard(args: argparse.Namespace) -> int:
    """Render a ranked leaderboard table (markdown or ``--json``)."""
    from robocurate.benchmark.leaderboard import load_leaderboard

    board = load_leaderboard(args.path)
    print(board.to_json() if args.json else board.to_markdown())
    return 0


def _default_name(submission_path: str) -> str:
    return Path(submission_path).stem


def _result_markdown(result: Any, *, name: str) -> str:
    """A compact markdown view of one scored submission (with the references and the caveat)."""
    mean = result.mean_loss_by_arm
    eff = result.submitted_vs_equal_n
    lines = [
        f"# Benchmark result — `{name}` ({result.submission_kind})",
        "",
        f"> {result.note}",
        "",
        f"- metric: `{result.metric}` (lower is better)",
        f"- episodes selected (train pool): {result.num_kept}",
        f"- seeds: {list(result.seeds)}",
        "",
        "| arm | mean held-out loss | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for arm in ("submitted", "equal_n_random", "full"):
        m = mean[arm]
        lines.append(
            f"| {arm} | {float(m['mean']):.4f} | "
            f"[{float(m['ci_low']):.4f}, {float(m['ci_high']):.4f}] |"
        )
    sep = "yes" if eff["separated"] else "no"
    lines += [
        "",
        f"**submitted vs equal-N random:** {float(eff['effect']):+.4f} "
        f"(95% CI [{float(eff['ci_low']):+.4f}, {float(eff['ci_high']):+.4f}]), "
        f"separated: {sep}.",
        "",
        "_A win is a NEGATIVE effect (the submission's held-out loss is lower than the "
        "equal-N random control's)._",
        "",
    ]
    return "\n".join(lines)


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
    p_curate.add_argument(
        "--recipe",
        help="load a saved JSON recipe and run it instead of --signals/--budget.",
    )
    p_curate.add_argument(
        "--save-recipe", help="write the run's config as a shareable JSON recipe to this path."
    )
    p_curate.add_argument(
        "--report-html", help="write a self-contained HTML curation report to this path."
    )
    p_curate.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="after a successful local write, push the curated output to this HF dataset repo.",
    )
    p_curate.add_argument(
        "--no-card", action="store_true", help="do not write a README.md dataset card."
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

    p_list = sub.add_parser("list-signals", help="list the available quality signals.")
    add_common(p_list)
    p_list.set_defaults(func=_cmd_list_signals)

    p_validate = sub.add_parser(
        "validate",
        aliases=["doctor"],
        help="diagnose a dataset's health (schema, structural defects, coverage). No write.",
    )
    p_validate.add_argument("dataset", help="path to a LeRobotDataset directory.")
    add_common(p_validate)
    p_validate.set_defaults(func=_cmd_validate)

    p_explain = sub.add_parser(
        "explain", help="explain one episode's kept/removed decision from a saved manifest."
    )
    p_explain.add_argument("path", help="path to a manifest or curated dataset.")
    p_explain.add_argument("episode_index", type=int, help="the source episode index to explain.")
    add_common(p_explain)
    p_explain.set_defaults(func=_cmd_explain)

    _add_benchmark_commands(sub, add_common)

    return parser


def _add_benchmark_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> None:
    """Wire the nested ``benchmark`` command group (init / run / leaderboard)."""
    p_bench = sub.add_parser(
        "benchmark",
        help="open-benchmark v0 — fixed pool + eval split + BC config; data is the submission.",
    )
    bench_sub = p_bench.add_subparsers(dest="benchmark_command", required=True)

    p_init = bench_sub.add_parser(
        "init", help="build a frozen benchmark spec (fixed eval split) from a pool dataset."
    )
    p_init.add_argument("pool", help="path to the pool LeRobotDataset directory.")
    p_init.add_argument("--out", required=True, help="destination spec JSON path.")
    p_init.add_argument(
        "--eval-frac", type=float, default=0.2, help="fraction held out for eval (default 0.2)."
    )
    add_common(p_init)
    p_init.set_defaults(func=_cmd_benchmark_init)

    p_run = bench_sub.add_parser(
        "run", help="score a submission (recipe or index-set) against a spec."
    )
    p_run.add_argument("submission", help="path to the submission JSON.")
    p_run.add_argument("--spec", required=True, help="path to the benchmark spec JSON.")
    p_run.add_argument("--pool", help="pool dataset path (defaults to the spec's pool dataset_id).")
    p_run.add_argument("--leaderboard", help="append the result to this leaderboard JSON.")
    p_run.add_argument("--name", help="submission name on the leaderboard (default: file stem).")
    p_run.add_argument(
        "--created-utc", help="timestamp recorded with the leaderboard entry (reproducible)."
    )
    add_common(p_run)
    p_run.set_defaults(func=_cmd_benchmark_run)

    p_lb = bench_sub.add_parser("leaderboard", help="render a ranked leaderboard table.")
    p_lb.add_argument("path", help="path to the leaderboard JSON.")
    add_common(p_lb)
    p_lb.set_defaults(func=_cmd_benchmark_leaderboard)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = args.func
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
