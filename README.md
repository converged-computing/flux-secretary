# flux-secretary

[![PyPI - Version](https://img.shields.io/pypi/v/flux-secretary)](https://badge.fury.io/py/flux-secretary)
[![DOI](https://zenodo.org/badge/1319769497.svg)](https://doi.org/10.5281/zenodo.21757790)

Runs a command inside a Flux allocation, correctly.

![https://github.com/converged-computing/flux-secretary/blob/main/img/flux-secretary-small.png](https://github.com/converged-computing/flux-secretary/blob/main/img/flux-secretary-small.png)

A scheduler decides *where* a job goes and *how big* it is. It cannot decide how
to launch it (e.g., the ranks, mapping, flags). Those depend on what the runtime
actually has, and that is only observable from inside the allocation.

flux-secretary runs as the workload's entrypoint, asks Flux what it was given,
assembles the submit, monitors it, and retries the launch until it works. It
exits with the job's return code so the outcome propagates back to the scheduler.
Built on [behalf](https://pypi.org/project/behalf/).

## Install

```bash
pip install flux-secretary[aws]     # or [claude], [gemini]
```

## Use

```bash
flux-secretary run -- osu_allreduce -m 8:1048576 -i 1000 -f
```

In a Flux Operator MiniCluster, set `launcher: true` so the operator does not wrap
the command in its own `flux submit`, and make flux-secretary the command:

```yaml
spec:
  size: 5
  tasks: 0
  launcher: true
  containers:
    - image: ghcr.io/example/app:latest
      command: flux python -m fluxsecretary.cli run -- osu_allreduce -m 8:1048576
      commands:
        pre: flux python -m pip install --user flux-secretary[aws]
      volumes:
        agent-token:
          secretName: flux-secretary-token
          path: /etc/flux-secretary
```

Attempts defaults to 4.

## Output

Every line is prefixed `FLUXSEC` and is `key=value`, and the last line is the whole
transcript as JSON. Ultimately this could be given back to another agent to parse (and decide
if a different submission is needed).

```console
FLUXSEC === flux-secretary ===
FLUXSEC command argv='osu_allreduce -m 8:1048576' nodes=5
FLUXSEC mode mode=deterministic token=none attempts_max=4
FLUXSEC resources nodes=5 cores=12 gpus=0 source='resource list'
FLUXSEC attempt status=failed rc=1 reason=alloc-unsatisfiable nodes=5 tasks=20 n=1
FLUXSEC attempt status=ok rc=0 jobid=f1 nodes=5 tasks=12 why='one rank per core' n=2
FLUXSEC result status=ok rc=0 attempts=2 jobid=f1 wall_s=44.1
FLUXSEC json {"status":"ok","attempts":[...],"job":{"runtime_s":41.2,...}}
```
```bash
kubectl logs <pod> | grep '^FLUXSEC json ' | cut -d' ' -f3- | jq .
```

## Tests

```bash
python3 tests/test_unit.py                                        # launch ladder only
flux start --test-size=4 flux python tests/test_integration.py    # against a real broker
```

## Requirements

You must be able to import flux in Python.

## License

HPCIC DevTools is distributed under the terms of the MIT license.
All new contributions must be made under this license.

See [LICENSE](https://github.com/converged-computing/cloud-select/blob/main/LICENSE),
[COPYRIGHT](https://github.com/converged-computing/cloud-select/blob/main/COPYRIGHT), and
[NOTICE](https://github.com/converged-computing/cloud-select/blob/main/NOTICE) for details.

SPDX-License-Identifier: (MIT)

LLNL-CODE- 842614
