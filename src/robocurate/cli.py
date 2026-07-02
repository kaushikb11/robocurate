"""Command-line interface — the surface shape for the v1 command set.

Commands mirror the Python API: ``score`` (report only, no write), ``curate`` (select +
write a new dataset + manifest), ``baseline`` (emit the equal-N random control), ``report``
(render a scorecard from a run), ``diff`` (raw vs curated), ``list-signals`` (the available
quality signals), ``validate``/``doctor`` (a read-only dataset health check), ``explain``
(one episode's kept/removed decision from a saved manifest), and ``rank`` (a read-only
"worst N episodes" report naming the signals responsible). Each accepts ``--seed`` and
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
from robocurate.curator import (
    ON_ERROR_ABORT,
    ON_ERROR_QUARANTINE,
    Budget,
    Curator,
    SelectionMode,
)
from robocurate.scorecard import Scorecard
from robocurate.signals.base import SignalContext

if TYPE_CHECKING:
    from robocurate.curator import CurationResult
    from robocurate.signals.base import Signal, SignalSpec, TrajectoryScore


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
    resolved = _resolve_signals(_split_csv(args.signals))
    reader = _open_pool_reader(args.dataset, include_videos=_needs_video(resolved))
    curator = Curator(resolved, seed=args.seed, on_error=args.on_error or ON_ERROR_ABORT)
    result = curator.run(reader)
    card = result.scorecard()
    print(card.to_json() if args.json else card.to_markdown())
    return 0


def _read_episode_index_list(path: str) -> list[int]:
    """Read a user episode list: a JSON array of indices, or ``{"episode_indices": [...]}``.

    The dict form is what ``rank --out-flags`` writes, so a review loop round-trips without
    editing; the bare-array form is the lowest-friction hand-written input.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read episode list {path}: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("episode_indices")
    if not isinstance(data, list) or not all(isinstance(i, int) for i in data):
        raise SystemExit(
            f"episode list {path} must be a JSON array of integer episode indices, or an "
            'object of the form {"episode_indices": [...]}'
        )
    return [int(i) for i in data]


def _curate_curator(args: argparse.Namespace) -> Curator:
    """Build the curator for a ``curate`` run: from a recipe, or from --signals/--budget.

    A recipe is mutually exclusive with --signals/--budget/--drop-list/--keep-list: a recipe
    already fixes the full config (combiner, budget, selection, gate, seed, episode lists), so
    mixing the two would be ambiguous.
    """
    from robocurate.recipe import load_recipe

    if args.recipe is not None:
        if (
            args.signals
            or args.budget is not None
            or args.drop_list is not None
            or args.keep_list is not None
        ):
            raise SystemExit(
                "--recipe is mutually exclusive with --signals/--budget/--drop-list/"
                "--keep-list: a recipe already fixes the full curation config. Pass one or "
                "the other, not both."
            )
        curator = load_recipe(args.recipe)
        if args.on_error is not None:
            # An explicit --on-error overrides the recipe's policy: it is an operational
            # read-tolerance knob, not part of the selection logic, and the run's manifest
            # records the value actually used.
            curator.on_error = args.on_error
        return curator
    if args.drop_list is not None and args.keep_list is not None:
        raise SystemExit(
            "--drop-list and --keep-list are mutually exclusive: a drop-list removes the "
            "listed episodes, a keep-list removes everything else."
        )
    drop = _read_episode_index_list(args.drop_list) if args.drop_list is not None else None
    keep = _read_episode_index_list(args.keep_list) if args.keep_list is not None else None
    names = _split_csv(args.signals)
    if not names and drop is None and keep is None:
        # Preserve the standard "which signals exist" error when there is nothing to do.
        _resolve_signals(names)
    # With a drop-/keep-list, signals are optional: a pure list-based removal (e.g. flags
    # exported from a review tool or `rank --out-flags`) is a valid curation on its own.
    signals_resolved = _resolve_signals(names) if names else []
    budget = Budget.fraction(args.budget) if args.budget is not None else None
    return Curator(
        signals_resolved,
        budget=budget,
        seed=args.seed,
        emit_baseline=not args.no_baseline,
        selection=SelectionMode(args.selection),
        coverage_quality_weight=args.coverage_quality_weight,
        drop_episode_indices=drop,
        keep_episode_indices=keep,
        on_error=args.on_error or ON_ERROR_ABORT,
    )


def _cmd_curate(args: argparse.Namespace) -> int:
    from robocurate.recipe import save_recipe

    curator = _curate_curator(args)
    reader = _open_pool_reader(args.dataset, include_videos=_needs_video(curator.signals))
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
    reader = _open_pool_reader(args.dataset)
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
    source = _open_pool_reader(args.source)
    curated = _open_pool_reader(args.curated)

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


# --------------------------------------------------------------------------------------
# inspect — score one episode in depth
# --------------------------------------------------------------------------------------

# Cheap, laptop-friendly signals run by ``inspect`` when --signals is not given. These are
# the Tier-0 signals that score a single low-dim episode meaningfully (no images, no GPU,
# no dataset-wide statistics required); a user can request any registered signal explicitly.
_INSPECT_DEFAULT_SIGNALS = (
    "jerk",
    "action_noise",
    "path_efficiency",
    "spectral_smoothness",
)


def _inspect_signal_context(seed: int, dataset_meta: Any) -> SignalContext:
    """Build a minimal CPU :class:`SignalContext` for running signals on one episode."""
    import logging

    from robocurate.metadata import ResourceProbe
    from robocurate.signals.base import InMemoryCache

    return SignalContext(
        seed=seed,
        device="cpu",
        cache=InMemoryCache(),
        resources=ResourceProbe(),
        dataset_meta=dataset_meta,
        logger=logging.getLogger("robocurate.inspect"),
    )


def _per_transition_summary(per_transition: Any, *, worst: int = 3) -> dict[str, Any]:
    """Summarize a ``(T,)`` per-transition trace: min/median/max + the worst few step indices.

    "Worst" is the highest-magnitude steps, since per-transition traces are cost/severity-like
    (larger = more notable); ties break by earliest step index for determinism.
    """
    import numpy as np

    arr = np.asarray(per_transition, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"length": int(arr.size), "min": None, "median": None, "max": None, "worst": []}
    order = sorted(range(arr.size), key=lambda i: (-abs(float(arr[i])), i))
    worst_steps = [
        {"step": int(i), "value": float(arr[i])} for i in order[:worst] if np.isfinite(arr[i])
    ]
    return {
        "length": int(arr.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
        "worst": worst_steps,
    }


def _inspect_one(signal: Signal, traj: Any, ctx: SignalContext) -> dict[str, Any]:
    """Fit (single-episode) then score ``traj`` with ``signal`` and build its descriptor."""
    signal.fit([traj], ctx)
    score = signal.score([traj], ctx)[0]
    entry: dict[str, Any] = {
        "signal": signal.spec.name,
        "value": None if score.skipped else float(score.value),
        "higher_is_better": score.higher_is_better,
        "skipped": score.skipped,
        "skip_reason": score.skip_reason,
        "diagnostics": dict(score.diagnostics),
    }
    if signal.spec.produces_per_transition and score.per_transition is not None:
        entry["per_transition"] = _per_transition_summary(score.per_transition)
    return entry


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Deep-dive one episode: run the requested signals and report value + per-transition trace."""
    names = _split_csv(args.signals) or list(_INSPECT_DEFAULT_SIGNALS)
    resolved = _resolve_signals(names)
    reader = _open_pool_reader(args.dataset, include_videos=_needs_video(resolved))
    try:
        traj = reader.read_episode(args.episode_index)
    except IndexError as exc:
        raise SystemExit(str(exc)) from exc
    ctx = _inspect_signal_context(args.seed, reader.meta)
    entries = [_inspect_one(sig, traj, ctx) for sig in resolved]

    payload: dict[str, Any] = {
        "dataset": str(args.dataset),
        "episode_index": traj.meta.episode_index,
        "fingerprint": traj.meta.fingerprint,
        "num_steps": traj.num_steps,
        "signals": entries,
    }
    if args.json:
        print(json.dumps(payload))
        return 0
    print(_inspect_markdown(payload))
    return 0


def _inspect_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Episode {payload['episode_index']} — `{payload['dataset']}`",
        "",
        f"- fingerprint: {str(payload['fingerprint'])[:12] or '(none)'}",
        f"- steps: {payload['num_steps']}",
        "",
        "## Signals",
        "",
    ]
    for entry in payload["signals"]:
        orient = "higher=better" if entry["higher_is_better"] else "lower=better"
        if entry["skipped"]:
            lines.append(f"### {entry['signal']} — skipped")
            lines.append(f"  reason: {entry['skip_reason'] or '(none given)'}")
            lines.append("")
            continue
        lines.append(f"### {entry['signal']}")
        lines.append(f"  value: {entry['value']:.4g} ({orient})")
        diagnostics = entry.get("diagnostics") or {}
        if diagnostics:
            shown = ", ".join(f"{k}={_fmt_diag(v)}" for k, v in diagnostics.items())
            lines.append(f"  diagnostics: {shown}")
        pt = entry.get("per_transition")
        if pt is not None:
            lines.append(
                f"  per-transition (T={pt['length']}): "
                f"min={_fmt_diag(pt['min'])} median={_fmt_diag(pt['median'])} "
                f"max={_fmt_diag(pt['max'])}"
            )
            if pt["worst"]:
                worst = ", ".join(f"step {w['step']}={_fmt_diag(w['value'])}" for w in pt["worst"])
                lines.append(f"  worst steps: {worst}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _fmt_diag(value: Any) -> str:
    """Format a diagnostic scalar compactly for markdown."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


# --------------------------------------------------------------------------------------
# rank — the "worst N episodes" report (read-only)
# --------------------------------------------------------------------------------------
#
# Once a dataset grows past a few dozen episodes, reviewing each one is impractical
# (huggingface/lerobot#3760). ``rank`` turns "watch 200 episodes" into "watch these 8": it
# scores every episode with cheap signals, combines them with the same machinery the curator
# uses, and surfaces the worst-scoring episodes with the signals responsible — a diagnosis
# starting point for "why doesn't my policy work". It never writes anything.


def _cmd_rank(args: argparse.Namespace) -> int:
    """Rank the worst episodes by combined keep-score and say which signals flagged each."""
    if args.worst < 1:
        raise SystemExit(f"--worst must be >= 1, got {args.worst}")
    names = _split_csv(args.signals) or list(_INSPECT_DEFAULT_SIGNALS)
    resolved = _resolve_signals(names)
    reader = _open_pool_reader(args.dataset, include_videos=_needs_video(resolved))
    # No budget: every episode is scored and none is selected away — this is a report, not a
    # curation. The baseline is skipped because there is no selection to compare against.
    curator = Curator(resolved, seed=args.seed, emit_baseline=False)
    result = curator.run(reader)
    payload = _rank_payload(result, dataset=str(args.dataset), worst=args.worst, seed=args.seed)
    if args.out_flags is not None:
        # The review loop's hand-off: rank → human review → `curate --drop-list <flags>`.
        # Deliberately just the ranked worst-N indices, in a shape curate accepts verbatim.
        flags = {"episode_indices": [e["episode_index"] for e in payload["worst"]]}
        Path(args.out_flags).write_text(json.dumps(flags, indent=2), encoding="utf-8")
    print(json.dumps(payload) if args.json else _rank_markdown(payload))
    return 0


def _rank_payload(result: CurationResult, *, dataset: str, worst: int, seed: int) -> dict[str, Any]:
    """Build the machine-readable rank report from a no-budget curation result.

    Episodes where *every* requested signal skipped are listed under ``unscored`` (with the
    skip reasons) and excluded from the ranking — the combiner would impute them to the
    neutral 0.5, which would silently rank them as mediocre rather than unknown.
    """
    matrix = result.score_matrix
    names = [spec.name for spec in matrix.signal_specs]
    normalized = {name: matrix.normalized_signal_scores(name) for name in names}

    ranked_positions: list[int] = []
    unscored: list[dict[str, Any]] = []
    for i, ref in enumerate(matrix.refs):
        scores = [matrix.scores.get((name, ref.fingerprint)) for name in names]
        if scores and all(s is None or s.skipped for s in scores):
            unscored.append(
                {
                    "episode_index": ref.episode_index,
                    "fingerprint": ref.fingerprint,
                    "skip_reason": _rank_skip_reason(scores),
                }
            )
        else:
            ranked_positions.append(i)

    # Worst first: ascending combined keep-score, ties broken by fingerprint — the same
    # deterministic tie-break the curator's selection uses (Invariant 3).
    keep = result.keep_scores
    ranked_positions.sort(key=lambda i: (keep[i], matrix.refs[i].fingerprint))
    shown = ranked_positions[:worst]

    entries: list[dict[str, Any]] = []
    for rank, i in enumerate(shown, start=1):
        ref = matrix.refs[i]
        per_signal: list[dict[str, Any]] = []
        for name in names:
            score = matrix.scores.get((name, ref.fingerprint))
            value = None if score is None or score.skipped else float(score.value)
            per_signal.append(
                {
                    "signal": name,
                    "value": value,
                    "normalized": float(normalized[name][i]),
                    "higher_is_better": True if score is None else score.higher_is_better,
                    "skipped": value is None,
                    "skip_reason": score.skip_reason if score is not None else None,
                }
            )
        worst_signals = _rank_worst_signals(per_signal)
        entries.append(
            {
                "rank": rank,
                "episode_index": ref.episode_index,
                "fingerprint": ref.fingerprint,
                "num_steps": ref.num_steps,
                "keep_score": float(keep[i]),
                "worst_signals": worst_signals,
                "reason": _rank_reason(worst_signals),
                "signals": per_signal,
            }
        )

    return {
        "dataset": dataset,
        "signals": names,
        "seed": seed,
        "num_episodes": matrix.num_trajectories,
        "num_ranked": len(ranked_positions),
        "num_shown": len(entries),
        "worst": entries,
        "unscored": unscored,
    }


def _rank_skip_reason(scores: list[TrajectoryScore | None]) -> str:
    """Join the distinct skip reasons for an all-skipped episode (order-stable)."""
    reasons: list[str] = []
    for score in scores:
        reason = (
            score.skip_reason
            if score is not None and score.skip_reason
            else "signal produced no score"
        )
        if reason not in reasons:
            reasons.append(reason)
    return "; ".join(reasons)


def _rank_worst_signals(
    per_signal: list[dict[str, Any]], *, limit: int = 2
) -> list[dict[str, Any]]:
    """The one or two signals most responsible for a low keep-score.

    The worst (lowest normalized keep-score) non-skipped signal is always named; a second is
    added only when it is itself below the 0.5 neutral point — so every named signal actually
    flags the episode, and no line hides behind an unexplained combined number (Invariant 6).
    """
    scored = [s for s in per_signal if not s["skipped"]]
    scored.sort(key=lambda s: (s["normalized"], s["signal"]))
    worst = scored[:1]
    if len(scored) > 1 and scored[1]["normalized"] < 0.5:
        worst.append(scored[1])
    return [
        {
            "signal": s["signal"],
            "value": s["value"],
            "normalized": s["normalized"],
            "higher_is_better": s["higher_is_better"],
        }
        for s in worst[:limit]
    ]


def _rank_reason(worst_signals: list[dict[str, Any]]) -> str:
    """One honest line naming the signal(s) responsible for a low keep-score.

    The percentage is the min-max position within *this* dataset's range on that signal
    (from the same keep-oriented normalization the combiner uses) — a relative statement
    about this dataset, not a calibrated probability that the episode hurts training.
    """
    if not worst_signals:
        return "no signal produced a score"
    parts: list[str] = []
    for j, s in enumerate(worst_signals):
        toward_worst = 100.0 * (1.0 - float(s["normalized"]))
        frag = (
            f"{s['signal']} (raw {s['value']:.4g}; {toward_worst:.0f}% of the way to this "
            f"dataset's worst value)"
        )
        parts.append(("worst on " if j == 0 else "also low on ") + frag)
    return "; ".join(parts)


def _rank_markdown(payload: dict[str, Any]) -> str:
    shown = payload["num_shown"]
    total = payload["num_episodes"]
    lines = [
        f"# Worst episodes — `{payload['dataset']}`",
        "",
        f"Showing the {shown} lowest-scoring of {total} episodes, worst first, ranked by the "
        f"combined keep-score of: {', '.join(payload['signals'])} (seed {payload['seed']}).",
        "",
        "> These heuristic signals are diagnostics, not proof an episode hurts training: a low",
        '> rank means "watch this one first", not "delete it". Scores are normalized within',
        "> this dataset (a relative ranking), and any removal should be validated against an",
        "> equal-N random baseline before you trust it.",
        "",
    ]
    for entry in payload["worst"]:
        lines.append(
            f"{entry['rank']}. **episode {entry['episode_index']}** — "
            f"keep-score {entry['keep_score']:.3f} — {entry['reason']}"
        )
    if not payload["worst"]:
        lines.append("(no episode could be ranked)")
    if payload["unscored"]:
        lines += [
            "",
            "## Unscored episodes (excluded from the ranking)",
            "",
            "Every requested signal skipped these episodes, so they have no keep-score. They",
            "are reported here rather than silently ranked as neutral:",
            "",
        ]
        lines += [
            f"- episode {u['episode_index']}: {u['skip_reason']}" for u in payload["unscored"]
        ]
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------------------
# profile — dataset EDA
# --------------------------------------------------------------------------------------


def _cmd_profile(args: argparse.Namespace) -> int:
    """Profile a source dataset (EDA): shapes, features, success, tasks, diversity. No write."""
    from robocurate.dataset import Dataset
    from robocurate.profile import dataset_profile

    dataset = Dataset.from_lerobot(args.dataset)
    report = dataset_profile(dataset)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_markdown())
    return 0


# --------------------------------------------------------------------------------------
# compare — diff two curation runs
# --------------------------------------------------------------------------------------


def _load_manifest(path_str: str) -> dict[str, Any]:
    """Load a manifest JSON from a path or a curated-dataset directory (like ``report``)."""
    path = Path(path_str)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"no manifest found at {path_str} (expected a manifest.json file)")
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read manifest {path}: {exc}") from exc
    return data


def _kept_set(manifest: dict[str, Any]) -> set[int]:
    """Reconstruct the kept episode-index set from a manifest's decisions."""
    return {d["episode_index"] for d in manifest.get("decisions", []) if d.get("kept")}


def _signal_summary(manifest: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Per-signal min/median/max over the decision-level signal values (NaN-safe).

    Mirrors how the scorecard summarizes a manifest's per-signal distribution, so the
    comparison's deltas line up with what ``report`` would show for each side.
    """
    import numpy as np

    decisions = manifest.get("decisions", [])
    names = [s["name"] for s in manifest.get("signals", [])]
    out: dict[str, dict[str, float | None]] = {}
    for name in names:
        values = np.array(
            [d.get("signal_values", {}).get(name, np.nan) for d in decisions], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            out[name] = {
                "min": float(finite.min()),
                "median": float(np.median(finite)),
                "max": float(finite.max()),
            }
        else:
            out[name] = {"min": None, "median": None, "max": None}
    return out


def _cmd_compare(args: argparse.Namespace) -> int:
    """Diff two curation runs: kept-set sizes, Jaccard overlap, flips, per-signal deltas."""
    manifest_a = _load_manifest(args.manifest_a)
    manifest_b = _load_manifest(args.manifest_b)

    kept_a = _kept_set(manifest_a)
    kept_b = _kept_set(manifest_b)
    union = kept_a | kept_b
    intersection = kept_a & kept_b
    jaccard = (len(intersection) / len(union)) if union else 1.0

    only_a = sorted(kept_a - kept_b)  # kept in A, removed in B
    only_b = sorted(kept_b - kept_a)  # kept in B, removed in A
    num_flipped = len(only_a) + len(only_b)

    summary_a = _signal_summary(manifest_a)
    summary_b = _signal_summary(manifest_b)
    signal_deltas = _signal_deltas(summary_a, summary_b)

    payload: dict[str, Any] = {
        "a": {"path": args.manifest_a, "num_kept": len(kept_a)},
        "b": {"path": args.manifest_b, "num_kept": len(kept_b)},
        "jaccard": jaccard,
        "num_intersection": len(intersection),
        "num_union": len(union),
        "num_flipped": num_flipped,
        "kept_in_a_only": only_a,
        "kept_in_b_only": only_b,
        "signal_deltas": signal_deltas,
    }
    if args.json:
        print(json.dumps(payload))
        return 0
    print(_compare_markdown(payload))
    return 0


def _signal_deltas(
    summary_a: dict[str, dict[str, float | None]],
    summary_b: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    """Per-signal (B - A) deltas on min/median/max for signals present in both manifests."""
    shared = sorted(set(summary_a) & set(summary_b))
    deltas: dict[str, dict[str, float | None]] = {}
    for name in shared:
        a, b = summary_a[name], summary_b[name]
        per_stat: dict[str, float | None] = {}
        for stat in ("min", "median", "max"):
            av, bv = a[stat], b[stat]
            per_stat[stat] = None if av is None or bv is None else bv - av
        deltas[name] = per_stat
    return deltas


def _compare_markdown(payload: dict[str, Any]) -> str:
    a, b = payload["a"], payload["b"]
    lines = [
        "# Curation comparison",
        "",
        f"- A: `{a['path']}` — kept {a['num_kept']}",
        f"- B: `{b['path']}` — kept {b['num_kept']}",
        "",
        f"- Jaccard overlap of kept episodes: {payload['jaccard']:.4f} "
        f"({payload['num_intersection']}/{payload['num_union']})",
        f"- Episodes that flipped kept<->removed: {payload['num_flipped']}",
    ]
    only_a = payload["kept_in_a_only"]
    only_b = payload["kept_in_b_only"]
    if only_a:
        lines.append(f"  - kept in A but not B: {_fmt_index_list(only_a)}")
    if only_b:
        lines.append(f"  - kept in B but not A: {_fmt_index_list(only_b)}")
    deltas = payload["signal_deltas"]
    if deltas:
        lines += ["", "## Per-signal summary deltas (B - A)", ""]
        lines.append("| signal | Δmin | Δmedian | Δmax |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, d in deltas.items():
            lines.append(
                f"| {name} | {_fmt_delta(d['min'])} | {_fmt_delta(d['median'])} "
                f"| {_fmt_delta(d['max'])} |"
            )
    return "\n".join(lines)


def _fmt_index_list(indices: list[int], *, limit: int = 10) -> str:
    """Render a few episode indices, eliding the tail when long."""
    shown = ", ".join(str(i) for i in indices[:limit])
    if len(indices) > limit:
        shown += f", … (+{len(indices) - limit} more)"
    return shown


def _fmt_delta(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}"


# --------------------------------------------------------------------------------------
# verify — reproducibility check (Invariant 3, made user-facing)
# --------------------------------------------------------------------------------------


def _curator_from_manifest(manifest: dict[str, Any]) -> Curator:
    """Rebuild the run's :class:`Curator` from a saved manifest.

    The manifest records the exact signal specs that ran (under ``signals``) plus the resolved
    config (combiner / budget / selection / gate / seed). The recipe loader reconstructs signals
    from the combiner's weight keys, but a run whose combiner used default (unit) weights names
    no signals there. So we seed those weight keys from the recorded signal names at their
    implicit default weight of 1.0 (the same weight ``WeightedSum`` applies to an unlisted
    signal), then reuse :func:`curator_from_config` — reproducing the run faithfully (Invariant 3)
    through the public reconstruction path, with byte-identical weighting.
    """
    from robocurate.curator import CurationConfig
    from robocurate.recipe import curator_from_config

    config = CurationConfig.from_dict(manifest["config"])
    combiner = dict(config.combiner_dict)
    weights = dict(combiner.get("weights", {}))
    for s in manifest.get("signals", []):
        weights.setdefault(s["name"], 1.0)
    combiner["weights"] = weights

    augmented = CurationConfig(
        combiner_dict=combiner,
        budget=config.budget,
        seed=config.seed,
        emit_baseline=config.emit_baseline,
        selection=config.selection,
        gate_dict=config.gate_dict,
        batch_size=config.batch_size,
        drop_episode_indices=config.drop_episode_indices,
        keep_episode_indices=config.keep_episode_indices,
        on_error=config.on_error,
    )
    return curator_from_config(augmented)


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-run a saved manifest/recipe and assert the recomputed selection matches it."""
    from robocurate.recipe import load_recipe

    spec = _load_manifest_or_none(args.spec)

    if spec is not None and "decisions" in spec:
        # A full curation manifest: it records the exact signals that ran AND the per-episode
        # decisions to check the re-run against.
        curator = _curator_from_manifest(spec)
        expected_kept: list[int] | None = sorted(_kept_set(spec))
        expected_reasons: dict[int, str] | None = {
            int(d["episode_index"]): str(d.get("reason", "")) for d in spec.get("decisions", [])
        }
    else:
        # A recipe carries config but no recorded decisions to check against; re-running it is
        # still a useful determinism smoke-test, but there is nothing to verify equality with.
        curator = load_recipe(args.spec)
        expected_kept = None
        expected_reasons = None

    reader = _open_pool_reader(args.dataset, include_videos=_needs_video(curator.signals))
    result = curator.run(reader)
    recomputed_kept = sorted(result.kept_episode_indices)
    recomputed_reasons = {int(d.episode_index): d.reason for d in result.decisions}

    mismatches = _verify_mismatches(
        expected_kept, recomputed_kept, expected_reasons, recomputed_reasons
    )
    verified = expected_kept is not None and not mismatches

    payload: dict[str, Any] = {
        "dataset": str(args.dataset),
        "spec": str(args.spec),
        "verified": verified,
        "has_recorded_decisions": expected_kept is not None,
        "num_kept_recomputed": len(recomputed_kept),
        "mismatches": mismatches,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(_verify_markdown(payload))
    return 0 if verified else 1


def _load_manifest_or_none(path_str: str) -> dict[str, Any] | None:
    """Load JSON at ``path_str`` (or its ``manifest.json``); ``None`` if unreadable as JSON."""
    path = Path(path_str)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"no manifest or recipe found at {path_str}")
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"could not read {path}: {exc}") from exc
    return data


def _verify_mismatches(
    expected_kept: list[int] | None,
    recomputed_kept: list[int],
    expected_reasons: dict[int, str] | None,
    recomputed_reasons: dict[int, str],
) -> list[str]:
    """Build a precise list of where the re-run diverged from the recorded manifest."""
    if expected_kept is None or expected_reasons is None:
        return []
    mismatches: list[str] = []
    if expected_kept != recomputed_kept:
        added = sorted(set(recomputed_kept) - set(expected_kept))
        dropped = sorted(set(expected_kept) - set(recomputed_kept))
        if added:
            mismatches.append(f"newly kept (not in manifest): {_fmt_index_list(added)}")
        if dropped:
            mismatches.append(f"no longer kept (were in manifest): {_fmt_index_list(dropped)}")
    for episode, expected_reason in expected_reasons.items():
        recomputed_reason = recomputed_reasons.get(episode)
        if recomputed_reason is not None and recomputed_reason != expected_reason:
            mismatches.append(
                f"episode {episode} reason changed: {expected_reason!r} -> {recomputed_reason!r}"
            )
    return mismatches


def _verify_markdown(payload: dict[str, Any]) -> str:
    verified = payload["verified"]
    lines = [
        "# Reproducibility check",
        "",
        f"- dataset: `{payload['dataset']}`",
        f"- spec: `{payload['spec']}`",
        f"- **verified: {'true' if verified else 'false'}**",
    ]
    if not payload["has_recorded_decisions"]:
        lines.append(
            "  (the spec is a recipe with no recorded decisions to verify against; the run "
            "executed deterministically but there was nothing to compare to)"
        )
    if payload["mismatches"]:
        lines += ["", "## Mismatches", ""]
        lines += [f"- {m}" for m in payload["mismatches"]]
    elif verified:
        lines.append(
            f"  Re-running reproduced all {payload['num_kept_recomputed']} kept episodes and "
            "their recorded reasons exactly."
        )
    return "\n".join(lines)


def _open_pool_reader(path: str, *, include_videos: bool = False) -> Any:
    """Open a LeRobotDataset (local directory or Hub id) as a read-only reader.

    Every dataset-reading CLI command routes through here so that all of them accept both
    on-disk LeRobot layouts and ``namespace/name`` Hub ids; nothing outside this helper names
    a concrete reader class. ``include_videos`` controls whether a Hub download pulls the mp4
    shards — pass it only when a requested signal actually decodes frames.
    """
    from robocurate.dataset import Dataset

    return Dataset.from_lerobot(path, include_videos=include_videos).reader


def _needs_video(resolved: list[Signal]) -> bool:
    """Whether any requested signal needs decodable frames (drives the Hub video download)."""
    from robocurate.signals.base import REQUIRES_IMAGE

    return any(REQUIRES_IMAGE in sig.spec.requires for sig in resolved)


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

    def add_on_error(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--on-error",
            choices=[ON_ERROR_ABORT, ON_ERROR_QUARANTINE],
            default=None,
            dest="on_error",
            help=(
                "reading-error policy: 'abort' (default — never silently tolerate "
                "corruption) or 'quarantine' (record each unreadable episode as removed, "
                "exclude it from the baseline pool, and keep going)."
            ),
        )

    p_score = sub.add_parser("score", help="score a dataset and print a scorecard (no write).")
    p_score.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_score.add_argument("--signals", help="comma-separated signal names.")
    add_on_error(p_score)
    add_common(p_score)
    p_score.set_defaults(func=_cmd_score)

    p_curate = sub.add_parser("curate", help="select a subset and write a new dataset.")
    p_curate.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_curate.add_argument("--out", required=True, help="destination for the curated dataset.")
    p_curate.add_argument("--signals", help="comma-separated signal names.")
    p_curate.add_argument("--budget", type=float, help="fraction of episodes to keep (0-1].")
    add_on_error(p_curate)
    p_curate.add_argument(
        "--drop-list",
        help="JSON file of episode indices to remove unconditionally (a JSON array, or "
        '{"episode_indices": [...]} as written by rank --out-flags). Signals become optional: '
        "a pure list-based removal is a valid curation. Mutually exclusive with --keep-list.",
    )
    p_curate.add_argument(
        "--keep-list",
        help="JSON file of episode indices to restrict the pool to; everything else is "
        "removed (same formats as --drop-list). Mutually exclusive with --drop-list.",
    )
    p_curate.add_argument(
        "--selection",
        choices=["top_k", "greedy_dedup", "coverage"],
        default="top_k",
        help=(
            "selection mode: top_k (highest keep-score), greedy_dedup (one representative per "
            "near-duplicate cluster), or coverage (diverse, representative subset)."
        ),
    )
    p_curate.add_argument(
        "--coverage-quality-weight",
        type=float,
        default=0.0,
        help="for --selection coverage: weight on keep-score vs. pure diversity (default 0.0).",
    )
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

    p_inspect = sub.add_parser(
        "inspect", help="deep-dive one episode: run signals and show per-transition traces."
    )
    p_inspect.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_inspect.add_argument("episode_index", type=int, help="the episode index to inspect.")
    p_inspect.add_argument(
        "--signals",
        help="comma-separated signal names (default: the cheap Tier-0 signals).",
    )
    add_common(p_inspect)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_rank = sub.add_parser(
        "rank",
        help=(
            "rank the worst episodes by combined keep-score, naming the signals responsible "
            "(read-only; 'watch these 8', not 200)."
        ),
    )
    p_rank.add_argument("dataset", help="path to a LeRobotDataset directory (v2.1 or v3).")
    p_rank.add_argument(
        "--signals",
        help="comma-separated signal names (default: the cheap Tier-0 signals).",
    )
    p_rank.add_argument(
        "--worst",
        type=int,
        default=10,
        help="how many episodes to surface (default 10, capped at the dataset size).",
    )
    p_rank.add_argument(
        "--out-flags",
        help="also write the ranked episode indices as a flags JSON file "
        '({"episode_indices": [...]}) that curate --drop-list accepts — review, then remove.',
    )
    add_common(p_rank)
    p_rank.set_defaults(func=_cmd_rank)

    p_profile = sub.add_parser(
        "profile", help="exploratory data analysis of a dataset (shapes, features, diversity)."
    )
    p_profile.add_argument("dataset", help="path to a LeRobotDataset directory.")
    add_common(p_profile)
    p_profile.set_defaults(func=_cmd_profile)

    p_compare = sub.add_parser(
        "compare", help="diff two curation runs (kept-set overlap, flips, per-signal deltas)."
    )
    p_compare.add_argument("manifest_a", help="first manifest or curated dataset.")
    p_compare.add_argument("manifest_b", help="second manifest or curated dataset.")
    add_common(p_compare)
    p_compare.set_defaults(func=_cmd_compare)

    p_verify = sub.add_parser(
        "verify",
        help="re-run a saved manifest/recipe and confirm it reproduces (Invariant 3).",
    )
    p_verify.add_argument("dataset", help="path to a LeRobotDataset directory.")
    p_verify.add_argument("spec", help="path to a saved manifest.json or a recipe JSON.")
    add_common(p_verify)
    p_verify.set_defaults(func=_cmd_verify)

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
