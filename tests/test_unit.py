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


if __name__ == "__main__":
    test_ladder_orders_most_to_least_specific()
    test_ladder_never_exceeds_the_allocation()
    test_token_is_exported_where_the_sdk_looks()
    test_backend_auto_is_resolved_to_something_behalf_knows()
    test_agent_failure_never_ends_the_run()
    print("\nunit tests passed")
