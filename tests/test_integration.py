"""Tests against a real Flux instance
Run these under a broker, which starts one for the duration:

    flux start --test-size=4 flux python tests/test_integration.py

They assert against what Flux actually does rather than a stand in, which is the
only way to know the eventlog keys and exception shapes are right.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluxsecretary import flux as fluxio


def test_resources_are_read_from_the_broker():
    r = fluxio.resources()
    assert r["nodes"] > 0 and r["cores"] > 0, r
    assert "free" in r and "all" in r, r
    print(f"OK resources: {r['nodes']} nodes, {r['cores']} cores")


def test_successful_job_reports_status_and_timestamps():
    r = fluxio.resources()
    out = fluxio.submit_and_wait(["hostname"], nodes=r["nodes"], tasks=r["nodes"])
    assert out["rc"] == 0 and out["success"], out
    assert out["events"][-1] == "clean", out["events"]
    assert "finish" in out["events"]
    for key in ("t_submit", "t_run", "t_finish", "t_cleanup", "runtime"):
        assert key in out, f"{key} missing from {sorted(out)}"
    assert not out["exceptions"] and not out["fatal"]
    print(f"OK completed in {out['runtime']}s, events {out['events']}")


def test_unsatisfiable_request_has_a_fatal_alloc_exception_and_no_finish():
    r = fluxio.resources()
    out = fluxio.submit_and_wait(["hostname"], nodes=r["nodes"], tasks=r["cores"] * 100)
    assert out["rc"] != 0 and not out["success"], out
    assert "finish" not in out["events"], out["events"]
    exc = out["fatal"][0]
    assert exc["type"] == "alloc" and exc["severity"] == 0, exc
    print(
        f"OK unsatisfiable: {exc['type']}/{exc['severity']} {exc['note']!r}, "
        f"no finish event so rc falls back to {out['rc']}"
    )


def test_nonzero_exit_is_reported_without_an_exception():
    out = fluxio.submit_and_wait(["/bin/false"], nodes=1, tasks=1)
    assert out["rc"] == 1 and not out["success"], out
    assert not out["exceptions"], "a failing task is not a Flux exception"
    print("OK non zero exit read from the finish status")


def test_exit_code_conversion():
    assert fluxio.exit_code(0) == 0
    assert fluxio.exit_code(256) == 1
    print("OK wait status conversion")


if __name__ == "__main__":
    for fn in (
        test_resources_are_read_from_the_broker,
        test_successful_job_reports_status_and_timestamps,
        test_unsatisfiable_request_has_a_fatal_alloc_exception_and_no_finish,
        test_nonzero_exit_is_reported_without_an_exception,
        test_exit_code_conversion,
    ):
        fn()
    print("\nintegration tests passed")
