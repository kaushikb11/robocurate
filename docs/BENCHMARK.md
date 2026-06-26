# The open benchmark (v0) — "DataComp-for-robotics" scaffolding

> **Honest scope.** This is **v0 scaffolding plus a runnable proof on a synthetic dataset with a
> proxy metric** — *not* "the benchmark the field has adopted". The scoring metric (held-out BC
> loss) is a CPU proxy with a documented bias; the real pool, an unbiased rollout-success metric,
> and a public leaderboard are the funded next step (see [`ROADMAP.md`](ROADMAP.md), Rung 2).

## The idea: the data is the submission

Vision/LLM benchmarks (DataComp, DCBench) fix the *model* and let you compete on the *data*.
Robotics has only the inverse (fix the data, vary the model). RoboCurate's open benchmark flips
it the data-centric way:

- a **fixed pool** of episodes,
- a **fixed held-out eval split** (carved deterministically, never trainable),
- a **fixed BC training config**.

A **submission is a selection** of pool episodes. Two submissions differ *only* in which episodes
they pick — so any difference in the score is attributable to the selection, not to a different
model, split, or recipe.

## The protocol

1. **Spec** (`benchmark init`). `build_spec(reader, eval_frac=0.2, seed=0)` freezes the pool's
   content fingerprint, carves the held-out eval split with the same deterministic routine the
   held-out evaluator uses (`_split_indices`), records the complementary train pool, and pins the
   BC training config + seeds. The spec is shareable JSON.
2. **Submission.** A small JSON file, one of two kinds:
   - **index-set** — `{"kept_episode_indices": [...]}`. The lowest-friction way to submit any
     selection (a baseline, or output from an external tool).
   - **recipe** — a saved RoboCurate recipe (has `recipe_version`). Resolving it *runs the
     curator* on the pool and takes the kept episodes. The cheap heuristic signals need no GPU or
     torch, so a recipe submission resolves on a core laptop install.
3. **Run** (`benchmark run`). `run_submission(spec, submission, pool_reader)`:
   - resolves the selection and **restricts it to the train pool** (eval episodes are never
     trainable);
   - builds three arms — `submitted`, `equal_n_random` (a same-size seeded random draw from the
     train pool — the fair control), and `full` (the whole train pool, a reference);
   - trains a BC policy per arm per seed under the fixed config and measures **held-out BC loss**
     on the fixed eval split;
   - reports per-arm bootstrap means and the paired `submitted` vs `equal_n_random` effect.
4. **Leaderboard** (`benchmark leaderboard`). Append-only, ranked by mean held-out loss ascending.
   The table **always** shows the `equal_n_random` and `full` references, the effect-vs-baseline
   with its `separated` verdict, and the proxy-metric caveat.

**Lower loss is better, so a win is a NEGATIVE effect** vs the equal-N control.

## The fair comparison (Invariant 5)

Every run computes an **equal-N random control**: a subset of the train pool the *same size* as
the submission, drawn with a seeded RNG (`SeedSequence([master_seed, stream]).generate_state(1)`,
mirroring the curator's own equal-N baseline). This neutralises the dataset-size confound — the
first thing a reviewer attacks. The control and the eval split are both seeded, so the same spec +
submission + seeds produce a byte-identical result (Invariant 3).

## The honest caveat: held-out BC loss is a *proxy*

The v0 metric is held-out behavior-cloning loss — a CPU-only stand-in for the faithful downstream
metric, closed-loop rollout success. It carries a **documented coverage bias toward the random
control**:

> The eval split is a *uniform-random* sample of the pool, so a uniform-random training subset is
> distribution-matched to it, while a deliberately non-uniform (curated) subset is not. This biases
> the proxy **toward the random control** via coverage, independent of demo quality. A selection
> that loses here is a *yellow flag*, not proof of harm; the unbiased arbiter is closed-loop task
> success (rollout), which does not reward matching the held-out *demo* distribution.

See the full discussion in [`src/robocurate/experiment/heldout.py`](../src/robocurate/experiment/heldout.py).
The spec's `metric` field is a **seam** for a future `rollout_success` backend that replaces the
proxy with the unbiased arbiter without changing the submission protocol.

## Runnable proof

[`examples/benchmark_identity.py`](../examples/benchmark_identity.py) builds the synthetic identity
dataset (helpful episodes have `action ≈ observation`; a harmful minority have `action ≈ -obs`),
freezes a spec, and scores two index-set submissions — "helpful only" vs an equal-size set that
includes the harmful episodes. The helpful-only submission wins with a clearly lower held-out loss
and a negative, `separated` effect vs the equal-N random control:

```
uv run --extra policy python examples/benchmark_identity.py
```

## CLI

```
robocurate benchmark init <pool_path> --out spec.json [--eval-frac 0.2] [--seed 0]
robocurate benchmark run <submission.json> --spec spec.json [--leaderboard lb.json] [--name id] [--json]
robocurate benchmark leaderboard <lb.json> [--json]
```
