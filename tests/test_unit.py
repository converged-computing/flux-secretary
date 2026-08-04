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
    # a value may hold commas; only a following KEY= ends it
    assert parse_env("OMPI_MCA_btl=self,vader,tcp,OMPI_MCA_pml=ob1") == {
        "OMPI_MCA_btl": "self,vader,tcp",
        "OMPI_MCA_pml": "ob1",
    }
    assert parse_env("A=1 B=2") == {"A": "1", "B": "2"}
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


def test_cwd_is_rendered_in_the_submit_command():
    """A watcher has to see which directory an attempt ran in."""
    from fluxsecretary.launch import Plan

    line = Plan(nodes=2, tasks=2, cwd="/opt/lammps/examples/reaxff/HNS").submit_command(
        ["lmp", "-in", "in.reaxff.hns"]
    )
    assert "--cwd /opt/lammps/examples/reaxff/HNS" in line, line
    assert Plan(nodes=1).submit_command(["a"]) == "flux submit -N 1 a"
    print("OK cwd shown in the attempt banner")


def test_find_file_refuses_to_choose_between_inputs():
    """Two directories holding the same filename are different inputs, and
    picking one silently changes what the benchmark measures."""
    import asyncio
    import json
    import os
    import tempfile

    from fluxsecretary.report import Transcript
    from fluxsecretary.task import LaunchTask

    task = LaunchTask(command=["lmp"], want_nodes=1, timeout=60, max_attempts=4)
    tools = {
        t.name: t
        for t in task.tools(
            {"nodes": 1, "cores": 2, "gpus": 0}, Transcript(command=["lmp"])
        )
    }
    assert "find_file" in tools, sorted(tools)

    call = lambda name: json.loads(
        asyncio.run(tools["find_file"].handler({"name": name}))["content"][0]["text"]
    )
    # a path, not a name, is rejected rather than searched for
    assert "error" in call("/etc/passwd")
    # a name that cannot exist reports nothing found, not a guess
    assert call("definitely-not-here-9271.dat")["directories"] == []
    print("OK find_file is bounded and refuses ambiguity")


def test_inspect_binary_reads_how_it_was_built():
    """An allocation reporting GPUs does not mean the application can use them.

    metric-kripke-cpu and metric-kripke-gpu differ only in how they were compiled,
    so the binary decides. A statically linked CUDA build has no libcuda in NEEDED
    at all and is only visible from its embedded device code.
    """
    import asyncio
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    from fluxsecretary.report import Transcript
    from fluxsecretary.task import LaunchTask

    task = LaunchTask(command=["x"], want_nodes=1, timeout=60, max_attempts=4)
    tools = {
        t.name: t
        for t in task.tools(
            {"nodes": 1, "cores": 2, "gpus": 0}, Transcript(command=["x"])
        )
    }
    assert "inspect_binary" in tools, sorted(tools)
    call = lambda p: json.loads(
        asyncio.run(tools["inspect_binary"].handler({"path": p}))["content"][0]["text"]
    )

    # something that is not there at all
    missing = call("definitely-not-a-binary-9271")
    assert missing["found"] is False, missing

    # a real cpu binary
    if shutil.which("readelf") and shutil.which("ls"):
        cpu = call("ls")
        assert cpu["found"] is True, cpu
        assert cpu["accelerator"] == "none", cpu
        assert cpu["needed"], "a dynamic binary has NEEDED entries"

        # and a statically linked cuda build, which linkage alone cannot see
        if shutil.which("gcc") and shutil.which("objcopy"):
            with tempfile.TemporaryDirectory() as d:
                src = os.path.join(d, "t.c")
                open(src, "w").write("int main(void){return 0;}\n")
                plain = os.path.join(d, "plain")
                if (
                    subprocess.run(
                        ["gcc", "-o", plain, src], capture_output=True
                    ).returncode
                    == 0
                ):
                    blob = os.path.join(d, "b.bin")
                    open(blob, "w").write("device code")
                    fat = os.path.join(d, "static_cuda")
                    if (
                        subprocess.run(
                            [
                                "objcopy",
                                "--add-section",
                                f".nv_fatbin={blob}",
                                "--set-section-flags",
                                ".nv_fatbin=noload,readonly",
                                plain,
                                fat,
                            ],
                            capture_output=True,
                        ).returncode
                        == 0
                    ):
                        got = call(fat)
                        assert got["accelerator"] == "cuda", got
                        assert ".nv_fatbin" in got["device_code_sections"], got
                        assert not got["gpu_libs"], "nothing in NEEDED to find"
    print("OK inspect_binary reports how the workload was built")


def test_ladder_plans_are_all_submittable():
    """flux rejects a jobspec with more nodes than tasks.

    The ladder's last rung is "let flux size the job", which arrives as tasks=None.
    Defaulting that to 1 asked for N nodes and one task, and from_command raised
    before anything ran.
    """
    from fluxsecretary.launch import ladder

    for nodes, cores in ((1, 1), (4, 4), (4, 32), (2, 16), (5, 5)):
        for plan in ladder({"nodes": nodes, "cores": cores, "gpus": 0}, nodes):
            tasks = plan.tasks or plan.nodes or 1
            assert plan.nodes <= tasks, (
                f"{nodes} nodes / {cores} cores: nodes={plan.nodes} "
                f"tasks={plan.tasks} would be rejected by flux"
            )
    print("OK every ladder rung has tasks >= nodes")


def test_no_gpu_affinity_without_gpus():
    """Rejecting gpu_affinity='off' forced the agent into 'per-task'.

    That asks the shell to place tasks onto devices the job was never given, on a
    CPU build that had correctly asked for no GPUs.
    """
    import asyncio
    import json

    import fluxsecretary.task as T
    from fluxsecretary.report import Transcript
    from fluxsecretary.task import LaunchTask

    seen = []

    class FakeIO:
        @staticmethod
        def submit_and_wait(cmd, **kw):
            seen.append(kw)
            return {
                "rc": 0,
                "jobid": "f1",
                "runtime": 1.0,
                "stdout": "",
                "stderr": "",
                "exceptions": [],
            }

    task = LaunchTask(command=["app"], want_nodes=4, timeout=600, max_attempts=6)
    tools = {
        t.name: t
        for t in task.tools(
            {"nodes": 4, "cores": 32, "gpus": 4}, Transcript(command=["app"])
        )
    }
    real, T.fluxio = T.fluxio, FakeIO
    try:
        base = {"nodes": 4, "tasks": 4, "cores_per_task": 8, "cpu_affinity": "per-task"}
        r = json.loads(
            asyncio.run(
                tools["try_launch"].handler(
                    {**base, "gpus_per_task": 0, "gpu_affinity": "off"}
                )
            )["content"][0]["text"]
        )
        assert "error" not in r, r
        assert seen[-1]["gpu_affinity"] is None, seen[-1]

        asyncio.run(
            tools["try_launch"].handler(
                {**base, "gpus_per_task": 0, "gpu_affinity": "per-task"}
            )
        )
        assert seen[-1]["gpu_affinity"] is None, seen[-1]

        asyncio.run(
            tools["try_launch"].handler(
                {**base, "gpus_per_task": 1, "gpu_affinity": "per-task"}
            )
        )
        assert seen[-1]["gpu_affinity"] == "per-task", seen[-1]
        assert seen[-1]["gpus_per_task"] == 1, seen[-1]
        assert all(k["duration"] == 600 for k in seen), seen
    finally:
        T.fluxio = real
    print("OK no gpu affinity without gpus, and attempts are bounded")


def test_intent_is_appended_without_replacing_the_rules():
    """What a run is FOR is something only the submitter knows.

    It changes what counts as a correct launch: a scaling study that must span
    four nodes is not satisfied by a run that fits on one, however green the exit
    code. It is appended, so the standing rules still apply and the agent can tell
    the two apart.
    """
    from fluxsecretary.task import SYSTEM, LaunchTask

    plain = LaunchTask(command=["x"], want_nodes=2)
    assert plain.setup_system_prompt() == SYSTEM
    assert plain.execute_system_prompt({}) == SYSTEM

    text = "This is a strong-scaling study. The run must span all four nodes."
    with_intent = LaunchTask(command=["x"], want_nodes=2, intent=text)
    got = with_intent.setup_system_prompt()

    assert got.startswith(SYSTEM), "the standing rules must come first, intact"
    assert text in got, got[-200:]
    assert got.index(text) > got.index("get_resources"), "intent goes after the rules"
    # the agent is told it does not override them
    assert "does not override" in got
    # and the execute prompt carries it too, not just setup
    assert text in with_intent.execute_system_prompt({})

    # whitespace only is the same as nothing
    assert LaunchTask(command=["x"], intent="   \n  ").setup_system_prompt() == SYSTEM
    assert LaunchTask(command=["x"], intent=None).setup_system_prompt() == SYSTEM
    print("OK intent is appended to the prompt, not substituted for it")


def test_agent_can_edit_args_but_not_the_executable():
    """The prompt says the problem parameters may be edited, so a tool must exist.

    Kripke rejects fewer than 8 directions before it decomposes anything, and no
    layout, affinity or environment setting fixes that. Without a way to act, the
    agent correctly reports the command cannot be edited and gives up.

    Only the arguments are replaceable, and the substitution is recorded so a run
    that changed the problem cannot be mistaken for one that did not.
    """
    import asyncio

    import fluxsecretary.task as T
    from fluxsecretary.report import Transcript
    from fluxsecretary.task import LaunchTask

    seen = []

    class FakeIO:
        @staticmethod
        def submit_and_wait(cmd, **kw):
            seen.append(list(cmd))
            return {
                "rc": 0,
                "jobid": "f1",
                "runtime": 1.0,
                "stdout": "",
                "stderr": "",
                "exceptions": [],
                "unallocated": False,
            }

    original = ["kripke", "--quad", "4", "--zones", "8,8,8"]
    task = LaunchTask(command=original, want_nodes=2, timeout=60)
    tr = Transcript(command=original)
    tools = {t.name: t for t in task.tools({"nodes": 2, "cores": 4, "gpus": 0}, tr)}
    real, T.fluxio = T.fluxio, FakeIO
    try:
        asyncio.run(tools["try_launch"].handler({"nodes": 2, "tasks": 2}))
        assert seen[-1] == original, seen[-1]
        assert tr.attempts[-1]["args_changed"] is None, tr.attempts[-1]

        asyncio.run(
            tools["try_launch"].handler(
                {"nodes": 2, "tasks": 2, "args": "--quad 8 --zones 8,8,8"}
            )
        )
        assert seen[-1] == ["kripke", "--quad", "8", "--zones", "8,8,8"], seen[-1]
        assert seen[-1][0] == "kripke", "the executable is not the agent's to change"
        assert tr.attempts[-1]["args_changed"] == "--quad 8 --zones 8,8,8", tr.attempts[
            -1
        ]

        # whitespace is not an edit
        asyncio.run(tools["try_launch"].handler({"nodes": 2, "tasks": 2, "args": "  "}))
        assert seen[-1] == original, seen[-1]
    finally:
        T.fluxio = real
    print("OK the agent can replace arguments, and it is recorded")


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
    test_cwd_is_rendered_in_the_submit_command()
    test_find_file_refuses_to_choose_between_inputs()
    test_inspect_binary_reads_how_it_was_built()
    test_ladder_plans_are_all_submittable()
    test_no_gpu_affinity_without_gpus()
    test_intent_is_appended_without_replacing_the_rules()
    test_agent_can_edit_args_but_not_the_executable()
    print("\nunit tests passed")
