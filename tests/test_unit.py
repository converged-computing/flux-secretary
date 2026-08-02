"""Unit tests that need no Flux
Only the launch ladder is pure logic. Everything else talks to a broker and is
covered by test_integration.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluxsecretary.launch import ladder


def test_ladder_orders_most_to_least_specific():
    plans = ladder({"nodes": 5, "cores": 15}, 5)
    assert [p.tasks for p in plans][:3] == [15, 5, None]
    print("OK ladder is ordered")


def test_ladder_never_exceeds_the_allocation():
    assert ladder({"nodes": 5, "cores": 20}, 10)[0].nodes == 5
    print("OK ladder caps at the allocation")


def test_token_is_exported_where_the_sdk_looks():
    """A mounted file means nothing to boto3. The token has to be put back into
    the environment variable the backend authenticates from.
    """
    import os

    from fluxsecretary.token import TOKEN_ENV, export_token

    for backend, want in (
        ("aws", "AWS_BEARER_TOKEN_BEDROCK"),
        ("claude", "ANTHROPIC_API_KEY"),
    ):
        for v in set(TOKEN_ENV.values()):
            os.environ.pop(v, None)
        assert export_token("tok", backend) == want
        assert os.environ[want] == "tok"

    # backend not yet resolved: set every one we know rather than guess wrong
    for v in set(TOKEN_ENV.values()):
        os.environ.pop(v, None)
    assert export_token("tok", "auto") == "all"
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "tok"

    # an explicit override wins
    assert export_token("tok", "aws", "MY_VAR") == "MY_VAR"
    assert os.environ["MY_VAR"] == "tok"
    for v in list(TOKEN_ENV.values()) + ["MY_VAR"]:
        os.environ.pop(v, None)
    print("OK token exported for the backend's SDK")


def test_backend_auto_is_resolved_to_something_behalf_knows():
    """behalf accepts claude, gemini or aws. Passing "auto" straight through made
    it exit, which ended the run instead of falling back.
    """
    import os

    from fluxsecretary.token import TOKEN_ENV, resolve_backend

    for v in set(TOKEN_ENV.values()):
        os.environ.pop(v, None)
    assert resolve_backend("auto") == "", "no credential means no agent"
    assert resolve_backend("auto", "file:/etc/flux-secretary/token") == "aws"
    assert resolve_backend("claude") == "claude", "an explicit backend wins"

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "x"
    assert resolve_backend("auto") == "aws"
    os.environ.pop("AWS_BEARER_TOKEN_BEDROCK")
    print("OK auto resolves to a real backend")


def test_agent_failure_never_ends_the_run():
    """The fallback catches BaseException, not just Exception: behalf raises
    SystemExit for an unknown backend, and that must not end the run.
    """
    import inspect

    from fluxsecretary import cli

    assert "except BaseException" in inspect.getsource(
        cli.main
    ), "SystemExit from the agent path would otherwise kill the run"
    print("OK agent failure degrades to deterministic")


def test_plan_renders_the_flux_command():
    """A watcher needs to see what was actually submitted."""
    from fluxsecretary.launch import Plan

    assert Plan(nodes=5, tasks=20).submit_command(["app", "-x"]) == (
        "flux submit -N 5 -n 20 app -x"
    )
    assert Plan(nodes=5).submit_command(["app"]) == "flux submit -N 5 app"
    assert Plan(nodes=2, tasks=4, cores_per_task=2).submit_command(["a"]) == (
        "flux submit -N 2 -n 4 -c 2 a"
    )
    print("OK flux command rendered per attempt")


def test_failure_reported_to_the_agent_includes_stderr():
    """A program that aborts says why in stderr; the agent has to see it."""
    from fluxsecretary.task import tail

    assert tail("short") == "short"
    long = "line\n" * 5000
    t = tail(long)
    assert len(t) < len(long) and t.endswith("line\n")
    assert "truncated" in t
    print("OK agent sees the tail of stderr")


def test_ladder_never_shrinks_below_the_allocation():
    """Fewer nodes is a different job, not a different launch."""
    from fluxsecretary.launch import ladder

    plans = ladder({"nodes": 2, "cores": 4}, 3)
    assert all(p.nodes == 2 for p in plans), [p.as_fields() for p in plans]
    print("OK ladder holds the node count")



def test_environment_parsed_from_a_string_or_mapping():
    """Agents pass environment either way, and a bad one must say so."""
    from fluxsecretary.task import parse_env

    assert parse_env(None) == {}
    assert parse_env({"A": 1}) == {"A": "1"}
    assert parse_env("OMPI_MCA_pml=ob1,FI_PROVIDER=tcp") == {
        "OMPI_MCA_pml": "ob1",
        "FI_PROVIDER": "tcp",
    }
    try:
        parse_env("not-a-pair")
    except ValueError as e:
        assert "KEY=VALUE" in str(e)
    else:
        raise AssertionError("a malformed environment must be rejected")
    print("OK environment parsed")


def test_submit_command_shows_the_environment():
    """A watcher has to see the variables an attempt actually set."""
    from fluxsecretary.launch import Plan

    line = Plan(
        nodes=2, tasks=2, environment={"OMPI_MCA_pml": "ob1"}, cpu_affinity="per-task"
    ).submit_command(["app"])
    assert line.startswith("OMPI_MCA_pml=ob1 flux submit -N 2 -n 2"), line
    assert "-o cpu-affinity=per-task" in line, line
    print("OK environment shown in the attempt banner")


if __name__ == "__main__":
    test_ladder_orders_most_to_least_specific()
    test_ladder_never_exceeds_the_allocation()
    test_token_is_exported_where_the_sdk_looks()
    test_backend_auto_is_resolved_to_something_behalf_knows()
    test_agent_failure_never_ends_the_run()
    test_plan_renders_the_flux_command()
    test_failure_reported_to_the_agent_includes_stderr()
    test_ladder_never_shrinks_below_the_allocation()
    test_environment_parsed_from_a_string_or_mapping()
    test_submit_command_shows_the_environment()
    print("\nunit tests passed")
