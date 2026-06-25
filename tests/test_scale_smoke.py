"""Scale smoke test: a curation run completes and stays deterministic at ~5,000 episodes.

This is a coarse "does it fall over / is it still deterministic" check at a few-thousand-
episode scale, not a benchmark. It uses the in-memory identity dataset and a cheap CPU signal
(``jerk``) so it runs without a GPU or any heavy dependency.

Memory note: the dataset is held fully in RAM here (``InMemoryDatasetReader``), and the
curator's ``_score`` retains a lightweight ``TrajectoryRef`` per episode plus one
``TrajectoryScore`` per (signal, episode) — not the trajectory arrays — so the run's resident
footprint scales with episode *count*, not total frames. The real memory ceiling at much
larger scale is the in-memory source itself; a streaming on-disk reader removes that ceiling.
"""

from __future__ import annotations

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.curator import Budget, Curator, WeightedSum
from robocurate.experiment.synthetic import make_identity_experiment_dataset
from robocurate.signals import get as get_signal

SCALE_EPISODES = 5_000


def _scale_reader() -> InMemoryDatasetReader:
    # ~5k tiny episodes: a helpful majority + contradictory minority (content distinct so
    # fingerprints differ), short trajectories to keep RAM modest.
    return make_identity_experiment_dataset(
        num_helpful=SCALE_EPISODES - 500,
        num_harmful=500,
        num_steps=8,
        seed=0,
    )


def _curator(seed: int = 0) -> Curator:
    return Curator(
        [get_signal("jerk")],
        combiner=WeightedSum(),
        budget=Budget.fraction(0.5),
        seed=seed,
    )


def test_scale_run_completes() -> None:
    reader = _scale_reader()
    assert len(reader) == SCALE_EPISODES

    result = _curator(seed=7).run(reader)

    # The run produced one decision per episode and respected the 50% budget.
    assert len(result.decisions) == SCALE_EPISODES
    assert result.num_kept == SCALE_EPISODES // 2
    assert result.num_removed == SCALE_EPISODES - result.num_kept
    assert result.baseline is not None
    assert result.baseline.n == result.num_kept


def test_scale_run_is_deterministic() -> None:
    reader = _scale_reader()
    a = _curator(seed=7).run(reader)
    b = _curator(seed=7).run(reader)
    assert a.kept_episode_indices == b.kept_episode_indices
    assert a.removed_episode_indices == b.removed_episode_indices
    assert a.baseline is not None and b.baseline is not None
    assert a.baseline.kept_episode_indices == b.baseline.kept_episode_indices
