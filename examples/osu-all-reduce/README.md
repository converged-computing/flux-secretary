# osu-all-reduce

An OSU all-reduce benchmark launched by flux-secretary, asking for more nodes
than the cluster has. The secretary reads what the allocation actually holds and
sizes the launch to it.

## Create a cluster

Make a cluster. I like Google Cloud.

```bash
gcloud container clusters create sched-gke-cpu --project <project> --zone us-central1-a --machine-type e2-standard-4 --num-nodes 2
gcloud container clusters get-credentials sched-gke-cpu --project <project> --zone us-central1-a
```

Install the Flux Operator:

```bash
kubectl --context sched-gke-cpu apply -f https://raw.githubusercontent.com/flux-framework/flux-operator/main/examples/dist/flux-operator.yaml
kubectl --context sched-gke-cpu -n operator-system rollout status deploy/operator-controller-manager --timeout=5m
```

## Give the cluster a token

The secretary runs inside the MiniCluster, so the credential has to exist in the cluster. The key must be named `token`: it is mounted at `/etc/flux-secretary/token`, which is where the secretary looks.

```bash
# Export your token to your environment
export AWS_BEARER_TOKEN_BEDROCK=...

# Create a secret from it.
kubectl --context sched-gke-cpu create secret generic flux-secretary-token --from-literal=token="$AWS_BEARER_TOKEN_BEDROCK"
```

You can also use `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`. Importantly, I chose this design because I do not want to put credentials in a system scheduler or tool. Without a token the job will still run, but just the command that you give it. Don't get it wrong!

## Run it

```bash
kubectl --context sched-gke-cpu apply -f minicluster.yaml
kubectl --context sched-gke-cpu get pods -w
```

Follow the broker, which is rank 0:

```bash
kubectl --context sched-gke-cpu logs -f job/osu-all-reduce --tail=100
```

## What to look for

Every line the secretary prints starts with `FLUXSEC` so you can parse it after. We install the Flux Secretary on the fly because I don't want to maintain another set of Flux views. Arguably we could add it there.

```
FLUXSEC mode mode=agent token=file:/etc/flux-secretary/token backend=aws
FLUXSEC resources nodes=2 cores=4 gpus=0 free_nodes=2 source=flux.resource.resource_list
FLUXSEC === attempt 1: flux submit -N 2 -n 4 /opt/osu-benchmark/... ===
FLUXSEC attempt status=ok rc=0 jobid=... runtime_s=41.2 nodes=2 tasks=4
FLUXSEC === attempt 1 stdout ===
# OSU MPI Allreduce Latency Test
...
FLUXSEC result status=ok rc=0 attempts=1 jobid=... mode=agent
FLUXSEC json {...}
```

`resources` is what Flux reported, not what was requested: the manifest asks for
3 nodes and the cluster has 2. `attempt` lines show each launch tried, with the
application's own output between markers. The last line is the whole transcript
as JSON, including timestamps from the job eventlog:

```bash
kubectl --context sched-gke-cpu logs job/osu-all-reduce \
    | grep '^FLUXSEC json ' | cut -d' ' -f3- | jq .
```

The application's output is unprefixed, so this gives just the benchmark:

```bash
kubectl --context sched-gke-cpu logs job/osu-all-reduce | grep -v '^FLUXSEC'
```

## Clean up

```bash
kubectl --context sched-gke-cpu delete minicluster osu-all-reduce
gcloud container clusters delete sched-gke-cpu --zone us-central1-a
```
