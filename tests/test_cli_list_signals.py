"""End-to-end tests for the ``list-signals`` CLI subcommand.

The command lists every loadable signal with its spec fields plus the optional extra it
needs, and surfaces any signal whose entry point failed to import. It runs against the
real registry (no fixtures), so the built-in Tier-0 signals must appear.
"""

from __future__ import annotations

import json

from robocurate.cli import main


def test_list_signals_markdown_lists_builtin_signals(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["list-signals"])
    assert rc == 0
    out = capsys.readouterr().out

    assert "Available signals" in out
    # The cheap built-in signals are always available (no optional extra).
    assert "jerk" in out
    assert "action_noise" in out
    # Tier-0 signals advertise no install extra.
    assert "none (Tier-0, CPU)" in out


def test_list_signals_json_is_machine_readable(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["list-signals", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)

    assert "available" in data and "unavailable" in data
    names = {entry["name"] for entry in data["available"]}
    assert {"jerk", "action_noise"} <= names

    by_name = {entry["name"] for entry in data["available"]}
    jerk = next(e for e in data["available"] if e["name"] == "jerk")
    # Each entry carries the spec fields the CLI promises.
    assert set(jerk) == {"name", "description", "cost_tier", "requires", "deterministic", "extra"}
    # A Tier-0 cheap signal needs no optional extra.
    assert jerk["cost_tier"] == "TIER0_CPU"
    assert jerk["extra"] is None
    assert "action_noise" in by_name
