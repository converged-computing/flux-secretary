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


def test_application_output_is_captured():
    """The job's own stdout and stderr come back with the result."""
    out = fluxio.submit_and_wait(
        ["sh", "-c", "echo from-the-app; echo from-stderr >&2"], nodes=1, tasks=1
    )
    assert out["rc"] == 0, out
    assert "from-the-app" in out["stdout"], out["stdout"]
    assert "from-stderr" in out["stderr"], out["stderr"]
    print("OK application output captured")


def test_intent_flags_reach_the_transcript():
    """--intent and --intent-file combine, and a missing file is an error."""
    import contextlib
    import io
    import tempfile
    import os

    from fluxsecretary.cli import main

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "intent.txt")
        open(path, "w").write("Prefer a layout that spans every node.\n")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(
                [
                    "run",
                    "--nodes",
                    "1",
                    "--model",
                    "none",
                    "--intent",
                    "From the flag.",
                    "--intent-file",
                    path,
                    "--",
                    "sh",
                    "-c",
                    "echo hi",
                ]
            )
        out = buf.getvalue()
        assert rc == 0, out
        assert "From the flag." in out, out
        assert "spans every node" in out, out
        assert "intent_chars=" in out, out

        # a file that cannot be read must fail, not run with no intent
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(
                [
                    "run",
                    "--nodes",
                    "1",
                    "--model",
                    "none",
                    "--intent-file",
                    os.path.join(d, "nope.txt"),
                    "--",
                    "sh",
                    "-c",
                    "echo hi",
                ]
            )
        assert rc == 2, (rc, buf.getvalue())
    print("OK intent flags combine, and a missing file is an error")


if __name__ == "__main__":
    for fn in (
        test_resources_are_read_from_the_broker,
        test_successful_job_reports_status_and_timestamps,
        test_unsatisfiable_request_has_a_fatal_alloc_exception_and_no_finish,
        test_nonzero_exit_is_reported_without_an_exception,
        test_exit_code_conversion,
        test_application_output_is_captured,
        test_intent_flags_reach_the_transcript,
    ):
        fn()
    print("\nintegration tests passed")
