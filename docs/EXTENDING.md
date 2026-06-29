# Extending RoboCurate

RoboCurate's core bet is that **the interesting work lives at the edges, not in the engine**.
A new quality signal or a new dataset format should plug in behind one small contract and
need *zero* changes to the core (Invariant 4). This guide shows how to write both, end to
end, and how to verify your signal honors the contract before you ship it.

Two extension points:

- **Signals** — score trajectories for quality. A signal is what decides "this episode is
  hurting your policy."
- **Adapters** — read a source dataset format into RoboCurate's canonical trajectory form.
  An adapter is how a new dataset format becomes curatable.

Everything below runs on a no-GPU laptop with the core install (NumPy + PyArrow); heavy
signals add their dependency behind an optional extra.

---

## Writing a custom signal

A signal implements the `Signal` protocol from `robocurate.signals.base`. There are exactly
three pieces:

1. a **`spec`** attribute (a `SignalSpec`) — what the signal advertises about itself;
2. a **`fit`** method — an optional one-shot pass over the whole dataset (a no-op for
   stateless heuristics);
3. a **`score`** method — scores a batch of trajectories, one `TrajectoryScore` per input.

The engine — not your signal — owns batching, scheduling, requirement gating, and caching.
Your signal stays small and never reaches into engine internals.

The complete, runnable version of the example below lives at
[`examples/custom_signal.py`](../examples/custom_signal.py).

### The spec

`SignalSpec` is the static contract your signal declares. The engine reads it to schedule
the signal, gate it on requirements, and label it in the scorecard:

```python
from robocurate.signals.base import CostTier, SignalSpec

self.spec = SignalSpec(
    name="action_range",            # unique; appears in scores, scorecard, manifest, cache keys
    version="0.1.0",                # bump to invalidate cached artifacts
    cost_tier=CostTier.TIER0_CPU,   # TIER0_CPU | TIER1_GPU | TIER2_GPU_HEAVY
    requires=frozenset({"action"}), # capability tokens; unmet -> a recorded skip, never a crash
    produces_per_transition=True,   # whether score() also emits a (T,) per-step array
    deterministic=True,             # must be True to join the seeded selection path (Invariant 3)
    description="Range (max - min) of per-step action magnitude (higher is more varied).",
)
```

**Cost tiers** tell the engine how expensive you are: `TIER0_CPU` (cheap, laptop-friendly —
jerk, action-noise), `TIER1_GPU` (a single GPU — a learned quality classifier), and
`TIER2_GPU_HEAVY` (GPU + time — CUPID-style influence; gated).

**Requirements** are free-form capability tokens — a feature key like `"action"` or
`"observation.state"`, or a well-known token like `REQUIRES_GPU` / `REQUIRES_SIM_STATE` /
`REQUIRES_IMAGE` (from `robocurate.signals.base`). New tokens can be added without a core
change. When a requirement is unmet, you record a **skip**, never raise.

### `fit` — the optional dataset pass

`fit` runs at most once per run, before any `score` call. A stateless heuristic just
returns:

```python
def fit(self, trajectories, ctx):
    return  # nothing to precompute
```

A learned signal does its one-shot training or precompute here and stashes the result in
`ctx.cache` (which is namespaced to your signal) for `score` to read. The built-in
`ActionNoise` signal is a worked example: it summarizes every trajectory in `fit` to compute
dataset-relative outlier statistics, then reads them back in `score`.

### `score` — one `TrajectoryScore` per input, in order

This is the heart of the contract. `score` takes a batch and returns **exactly one
`TrajectoryScore` per input trajectory, in the same order**:

```python
import numpy as np
from robocurate.signals.base import TrajectoryScore

def score(self, batch, ctx):
    return [self._score_one(traj) for traj in batch]

def _score_one(self, traj):
    fingerprint = traj.meta.fingerprint
    actions = traj.actions()
    if actions is None:
        # Unmet requirement -> a recorded skip, never an exception, never a silent drop.
        return TrajectoryScore.skip(
            self.spec.name, fingerprint,
            reason="no action feature to measure range over",
            higher_is_better=True,
        )

    flat = np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1)
    per_step = np.linalg.norm(flat, axis=1)           # a (T,) per-step magnitude
    value = float(per_step.max() - per_step.min())

    return TrajectoryScore(
        signal=self.spec.name,
        trajectory_fingerprint=fingerprint,
        value=value,
        higher_is_better=True,                         # orientation, so the curator never guesses
        per_transition=per_step.astype(np.float32),    # required iff produces_per_transition
        diagnostics={"max_magnitude": float(per_step.max())},  # free-form, surfaced in the scorecard
    )
```

`TrajectoryScore` (from `robocurate.signals.base`) carries:

- `value` — the trajectory-level scalar (each signal on its own raw scale; the curator
  normalizes across signals before combining).
- `higher_is_better` — orientation of `value`. A roughness score is lower-is-better; a
  quality score is higher-is-better. Set it so the curator never has to guess.
- `per_transition` — an optional `(T,)` per-step array. Present **iff** your spec sets
  `produces_per_transition`, and its length **must** equal `traj.num_steps`.
- `skipped` / `skip_reason` — a recorded skip (see below). A skipped score carries `NaN` and
  never silently becomes a removal.
- `diagnostics` — free-form per-trajectory diagnostics for the scorecard.

### Skip gracefully — never crash

A trajectory your signal can't score (missing feature, unmet requirement, too short, bad
timestamps) is a **recorded skip**, not an exception and not a silent removal. Use the
`TrajectoryScore.skip(...)` constructor — it sets `skipped=True`, attaches your reason, and
carries `NaN` as the value:

```python
return TrajectoryScore.skip(
    self.spec.name, fingerprint, reason="trajectory too short for this signal",
)
```

A signal that raises instead of skipping is a contract violation (and the contract-checker
flags it).

### Determinism (Invariant 3)

If your spec sets `deterministic=True`, two runs on the same `(batch, ctx, seed)` must
produce **identical** values. Pure NumPy heuristics are deterministic for free. If your
signal needs randomness, derive a seeded stream from `ctx.seed` — never use an unseeded RNG
on the selection path. Non-deterministic signals are allowed only in report-only scoring and
must set `deterministic=False` honestly.

### Registering your signal

Built-in and third-party signals register the same way: through the `robocurate.signals`
**entry-point group** in your package's `pyproject.toml`. This is what lets a community
signal be discovered without touching the RoboCurate core:

```toml
[project.entry-points."robocurate.signals"]
action_range = "my_package.signals:ActionRange"
```

The target is a zero-argument-constructible `Signal` factory (a class works). Once your
package is installed, `robocurate.signals.available()` lists it and
`robocurate.signals.get("action_range")` instantiates it. For a quick experiment or a test
you can also register programmatically:

```python
from robocurate import signals
signals.register("action_range", ActionRange)
sig = signals.get("action_range")
```

### Verify it: `check_signal_contract`

Before you ship, run the contract-checker. It runs a battery of structural and behavioral
checks against any `Signal` and returns a list of human-readable violations — **an empty
list means it passes**:

```python
from robocurate.signals import check_signal_contract

violations = check_signal_contract(ActionRange())
assert violations == []
```

With no arguments it builds its own tiny synthetic batch and CPU context, so it's a true
one-liner. It checks that your spec is well-formed, that `fit` and `score` run without
error, that `score` returns one score per input in order (matched by fingerprint), that
every score is either a skip-with-reason or a finite value, that `per_transition` matches
`produces_per_transition` and the trajectory length, and that a `deterministic` signal is
actually stable across runs.

Put it in your test suite with the assert-style wrapper, which raises an `AssertionError`
listing every violation:

```python
from robocurate.signals import assert_signal_contract

def test_my_signal_honors_the_contract():
    assert_signal_contract(MySignal())
```

You can also pass your own `trajectories=` and `ctx=` to exercise the signal against
representative data (e.g. a trajectory that *should* be skipped).

---

## Writing a custom adapter

An adapter reads a source dataset format into RoboCurate's canonical `Trajectory`. The read
side implements the `DatasetReader` protocol from `robocurate.adapters.base`. The protocol is
deliberately **read-only — it has no write method at all**, so there is physically no way to
mutate the source through it. That is how Invariant 1 (source data is read-only) becomes a
*type-level* guarantee, not just a convention.

A `DatasetReader` provides:

```python
class DatasetReader(Protocol):
    meta: DatasetMeta                          # dataset-level metadata

    def __len__(self) -> int: ...              # number of episodes
    def __iter__(self) -> Iterator[Trajectory]: ...   # lazy, episode-index order
    def read_episode(self, index: int) -> Trajectory: ...  # one episode by index
    def fingerprint(self) -> DatasetFingerprint: ...  # stable content hash of the source
```

Iteration is **lazy** — yield episodes one at a time so a dataset too large for RAM streams
episode-by-episode rather than loading whole. Open source files read-only (`"r"`) and never
write back.

### Building a `Trajectory`

Each episode becomes one `Trajectory`, which separates **schema** (what features exist) from
**data** (the arrays). The pieces (all from `robocurate.trajectory`):

- An **`EmbodimentSpec`** describing the observation/action space — a `FeatureSpec` per
  feature column, each with a `FeatureRole` (`ACTION`, `STATE`, `PROPRIO`, `REWARD`, `TIME`,
  `IMAGE`, `SUCCESS`, `EXTRA`), explicit `units`, and optional per-dim `names`. Meaning is
  always read from here, never inferred.
- A **`FeatureStore`** that materializes the `(T, *shape)` array for each feature key. Use
  the bundled `InMemoryFeatureStore` for small/synthetic data; real adapters supply a lazy,
  memory-mapped or decode-on-demand store so cheap signals never decode video.
- A **`TrajectoryMeta`** carrying the source dataset id, episode index, the embodiment, the
  content `fingerprint`, `num_steps`, the `source_format`, an optional `SuccessLabel`, and
  free-form `extra` metadata (task string, sim seed, ground-truth quality label, ...).

Compute the per-episode fingerprint with `fingerprint_arrays(columns)` — a stable content
hash over the raw feature bytes. Sharing this helper means a round-tripped trajectory keeps
its fingerprint, which is what links it to the manifest and powers dedup.

```python
from robocurate.trajectory import (
    EmbodimentSpec, FeatureRole, FeatureSpec, InMemoryFeatureStore,
    SuccessLabel, Trajectory, TrajectoryMeta, fingerprint_arrays,
)

columns = {"timestamp": t, "action": action, "observation.state": state}
meta = TrajectoryMeta(
    source_dataset_id="my_format/dataset",
    episode_index=i,
    embodiment=embodiment,
    fingerprint=fingerprint_arrays(columns),
    num_steps=t.shape[0],
    source_format="my_format_v1",
    success=SuccessLabel(value=True, source="demonstrator"),
)
trajectory = Trajectory(meta, InMemoryFeatureStore(columns))
```

The convention every signal relies on: the **leading axis of every feature is time (`T`)**,
and `T` equals `meta.num_steps` for every feature.

### Reference implementations

Read these alongside the protocol — they are the clearest worked examples:

- [`src/robocurate/adapters/hdf5.py`](../src/robocurate/adapters/hdf5.py) — a generic HDF5
  reader (schema-driven, lazy).
- [`src/robocurate/adapters/robomimic.py`](../src/robocurate/adapters/robomimic.py) — a
  robomimic HDF5 reader that also carries a ground-truth operator-skill tier through
  `meta.extra` — a good example of preserving source metadata losslessly.
- [`src/robocurate/adapters/lerobot.py`](../src/robocurate/adapters/lerobot.py) — the
  LeRobotDataset v2.1 reader (the native format).

Writing a curated dataset back out is the `DatasetWriter` side of the protocol; it always
writes a **new** dataset and validates it (schema + checksum + round-trip reload) before
reporting success. You only need a writer if you're adding a new *output* format — for a new
*source* format, a `DatasetReader` is enough, since RoboCurate emits curated datasets in the
LeRobotDataset format.

---

## Checklist before you open a PR

- [ ] Signal implements `spec` + `fit` + `score`; adapter implements the `DatasetReader`
      protocol (read-only).
- [ ] `check_signal_contract(MySignal())` returns `[]` (and it's asserted in a colocated
      test).
- [ ] Unmet requirements / missing features produce a recorded **skip**, never a raise.
- [ ] `deterministic=True` is honest — no unseeded RNG on the selection path.
- [ ] Heavy dependencies live behind an optional extra, with tests marked so the core-only
      CI run skips them (see `CONTRIBUTING.md`).
- [ ] Registered via the `robocurate.signals` entry-point group in `pyproject.toml`.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the dev loop and the full invariant list.
```
