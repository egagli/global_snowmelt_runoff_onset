# Embarrassingly-parallel array processing on GitHub Actions + Icechunk

*A portable pattern: free CI runners as a compute fleet, one transactional cloud
array store as both the output and the only source of truth about progress.
Distilled from the global snowmelt runoff onset pipeline (v10), where it
processed ~47,000 work units (~4,400 tiles × 11 years, reading ~60 TB of
satellite imagery selected from a 2,400 TB catalog) using only GitHub-hosted
runners — no cluster, no orchestrator, no database, and no compute bill on a
public repo. Code snippets are simplified from the real pipeline
(`processing/scripts/`, `global_snowmelt_runoff_onset/status.py`) and use
generic names so they port.*

## The idea in one paragraph

Cut a large gridded computation into **work units that write disjoint regions
of one cloud array store** (here: one tile × one year = one Zarr shard). Run
each unit as a GitHub Actions matrix job. Every unit that finishes makes **one
Icechunk commit** carrying machine-readable metadata; every unit that fails
makes **no commit at all**. Then the entire control plane collapses into one
function: *fold over the store's commit history*. What has a commit is done,
what doesn't is missing, and "dispatch the fleet" just means "list the missing
units and press the button again until the list is empty."

## Why this works

Three properties have to hold, and each maps to one technology choice:

1. **Concurrent writers never conflict** — because work units write disjoint
   regions, and the store's shards are aligned to those regions. Icechunk
   commits are transactional; its `ConflictDetector` rebases concurrent
   commits automatically when they touch disjoint chunks, which is *always*
   true here by construction. 100+ simultaneous jobs committing to one repo is
   routine.
2. **The ledger cannot drift from the data** — because the ledger *is* the
   data's commit history. There is no status CSV or database to get out of
   sync, no cleanup after crashes: a killed job leaves nothing behind but
   absence, and absence already means "re-dispatch me."
3. **The compute is elastic and effectively free** — GitHub-hosted runners
   (free for public repos) give ~16 GB RAM / 4 cores / tens of GB SSD per job,
   hundreds of jobs in parallel. The price is that runners are ephemeral,
   heterogeneous, and occasionally broken — which the commit-ledger design
   absorbs, because any failure costs only that unit's re-run.

## Architecture

```text
┌────────────────────┐   one-time    ┌─────────────────────────────────┐
│ init notebook/script│ ───────────▶ │ Icechunk repo on object storage │
│ metadata-only        │             │  Zarr v3, full-extent arrays,   │
│ template store       │             │  fill value everywhere,         │
└────────────────────┘              │  shard == work-unit region      │
                                    └────────────▲───────────┬────────┘
                                                 │ commits    │ ancestry
┌────────────────────┐  matrix JSON  ┌───────────┴──────┐    │
│ dispatch workflow   │ ────────────▶│ N worker jobs    │    │
│ (derives remaining  │              │ (one work unit   │    │
│  work from ancestry)│◀─────────────│  chunk each)     │    │
└────────────────────┘   re-dispatch └──────────────────┘    │
        ▲                until empty                          │
        └─────────────────── status fold ◀───────────────────┘
```

## The seven rules

### 1. Initialize the store as an empty template, sharded by work unit

Create the full-extent arrays once, up front, writing **metadata only**: a
global store of any size costs almost nothing until data lands. Choose the
shard shape to equal one work unit's region (here `(1 year, 2048, 2048)` px)
with smaller inner chunks for read granularity.

```python
import icechunk, xarray as xr

storage = icechunk.azure_storage(account=..., container=..., prefix=..., sas_token=...)
repo = icechunk.Repository.create(storage)          # or .open() thereafter
session = repo.writable_session("main")

# template_ds: full-extent dataset of the right dtypes/coords, values never computed
template_ds.to_zarr(
    session.store, zarr_format=3, consolidated=False, compute=False,   # metadata only
    encoding={
        var: {
            "shards": (1, TILE_DIM, TILE_DIM),      # shard == one work unit's region
            "chunks": (1, INNER_DIM, INNER_DIM),    # read granularity within a shard
            "fill_value": NODATA,                   # see warning below
        }
        for var in template_ds.data_vars
    },
)
session.commit("initialize empty template store")
```

Workers then write with `region=` bounds — never appends — so writes are
position-independent and order-independent:

```python
ds_unit.to_zarr(session.store, mode="r+", zarr_format=3, consolidated=False,
                region={"time": slice(t, t + 1),
                        "y": slice(r0, r0 + TILE_DIM),
                        "x": slice(c0, c0 + TILE_DIM)})
```

> ⚠️ Set the template's `fill_value` to your true nodata sentinel. If it
> defaults to 0, *absent* chunks read back as valid zeros and empty regions
> become indistinguishable from data. This bug cost us a store rebuild.

### 2. One commit per work unit, with machine-readable metadata

Icechunk commits accept a metadata dict alongside the human message. The
message is for humans; **all tooling parses only the metadata**:

```python
{
  "schema": 1,                       # version the schema itself
  "kind": "tile_year",               # or "tile_composite" for derived products
  "tile": [row, col], "water_year": 2019,
  "status": "data" | "empty",
  "empty_reason": "no_seasonal_snow" | "no_s1_data" | "no_valid_pixels",
  "stats": {"valid_px": ..., "n_scenes": ...},   # whatever your QA needs
  "missing_assets": [...],           # any inputs deliberately skipped (rule 6)
  "config_version": "v10",
  "duration_s": 123.4,
  "provenance": {"git_sha": ..., "runner": ..., "package_versions": ...}
}
```

Commit through a bounded retry that opens a **fresh session per attempt** —
sessions expire, conflicts rebase, and programming errors re-raise immediately
since retrying can't fix them:

```python
def commit_with_retry(repo, branch, write_fn, message, metadata, allow_empty=False):
    for attempt in range(COMMIT_MAX_TRIES):                    # bounded on purpose
        try:
            session = repo.writable_session(branch)
            write_fn(session)                                  # the region write, or no-op
            return session.commit(message, metadata=metadata,
                                  rebase_with=icechunk.ConflictDetector(),
                                  allow_empty=allow_empty)
        except (ValueError, KeyError, TypeError):
            raise                                              # schema/programming error
        except Exception as e:                                 # conflict, expired session, blip
            delay = min(60, 2 ** attempt) * random.uniform(0.5, 1.5)
            log.warning(f"commit attempt {attempt+1} failed ({e}); retry in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"commit failed after {COMMIT_MAX_TRIES} attempts")
```

### 3. Failure = no commit, absence = missing — the whole crash story

Workers never write partial results and never record failures. The worker
skeleton is just:

```python
for sub_unit in requested_sub_units:            # e.g. the years of one tile
    result = process(sub_unit)                  # any exception -> job exits nonzero,
    commit_with_retry(repo, "main",             #   nothing committed, unit stays missing
                      lambda s: write_region(s, result),
                      message, build_metadata(sub_unit, result))
```

A job that OOMs, times out, hits a network hole, or has its runner die
mid-flight leaves the ledger untouched, and the next status fold re-lists its
unit. This single convention replaces checkpointing, cleanup handlers, and
dead-letter queues. Corollary: make sub-units small enough that losing one is
cheap (we commit one year at a time inside a tile job precisely so a crash
loses ≤ 1 unit and committed years are never redone).

### 4. "Verified empty" is a result, and it carries a reason

Many units legitimately have nothing to compute. Record that as an **empty
commit** with a reason enum — otherwise the dispatcher retries barren units
forever. The critical discipline: an empty marker may only follow a
*successful* check, never a failed one — a transient upstream error recorded
as "empty" is a silent, durable lie.

```python
items = search_catalog_with_retries(unit)       # raises after N backoff attempts:
if len(items) == 0:                             #   a FAILED search never reaches here
    commit_with_retry(repo, "main", lambda s: None,   # no data written --
                      f"{unit}: empty (no_input_data)",
                      build_metadata(unit, status="empty", empty_reason="no_input_data"),
                      allow_empty=True)         # -- but the verdict is durable
    continue
```

### 5. Status and dispatch are one pure fold over the ancestry

```python
def get_status(repo, branch="main", as_of_snapshot=None):
    ancestry = (repo.ancestry(snapshot_id=as_of_snapshot) if as_of_snapshot
                else repo.ancestry(branch=branch))
    seen = {}
    for snapshot in ancestry:                       # newest -> oldest
        meta = snapshot.metadata or {}
        if "schema" not in meta:                    # init/maintenance commits
            continue
        key = (tuple(meta["tile"]), meta.get("water_year"))
        seen.setdefault(key, meta)                  # first seen == newest wins
    return seen

def get_remaining_work(repo, all_expected_units, today):
    seen = get_status(repo)
    return [u for u in all_expected_units
            if u not in seen and is_eligible(u, today)]        # eligibility: see below
```

- **Newest-wins** makes reprocessing free: recommitting a unit supersedes it,
  no deletion needed. (Do log a loud warning when a worker is about to
  supersede an existing commit — silent recomputes hide dispatch mistakes.)
- **Snapshot-pin the fold for a fleet run**: derive the work list once, pass
  the snapshot id through the workflow chain, and have every batch re-derive
  against that same snapshot — otherwise batch boundaries shift as commits
  land mid-run. Cost is negligible (~1 ms per commit walked; tens of
  thousands of commits ≈ well under a minute).
- Gate dispatch with **eligibility rules** when ground truth isn't final yet.
  We gate each hemisphere's water year until its season has fully elapsed —
  enforced in both the dispatcher *and* the worker, so an early dispatch can
  never commit a premature verdict:

```python
def is_eligible(unit, today, buffer_days=120):
    end = season_end(unit.water_year, unit.hemisphere)   # domain-specific
    return today >= end + timedelta(days=buffer_days)
# worker side: ineligible sub-units are skipped WITHOUT any commit --
# they stay 'missing' and flow into the work list once their date passes
```

### 6. When inputs are broken, skip loudly, record forever, and cap the damage

Real catalogs contain corpses (entries whose underlying files are gone). Don't
let one dead input fail a unit forever, and don't let a storage outage quietly
thin your data either:

```python
except LoadError:
    dead = probe_assets_yourself(failing_uris)        # only a confirmed 404 counts;
    #                                                   every transient falls through
    #                                                   to the ordinary bounded retry
    candidate = dead_so_far | dead
    max_by_fraction = max(1, math.ceil(MAX_DEAD_FRACTION * n_inputs))   # ceil + floor!
    if len(candidate) > MAX_DEAD_COUNT or len(candidate) > max_by_fraction:
        raise                                          # smells like an outage: fail LOUD
    retry_unit_without(candidate)                      # and record ids in commit
    #                                                    metadata["missing_assets"]
```

The `ceil`-with-floor-of-one matters: a 5% cap that rounds down forbids
dropping even one input from a 13-input unit — which deadlocked our sparse
Arctic tiles (1/13 = 7.7% > 5%) until fixed. Recording the dropped ids in
metadata keeps thinned units distinguishable from whole ones, and findable if
the upstream archive is ever repaired.

### 7. Derived products track staleness by commit order

Cross-unit aggregates (our multi-year composites) get their own commit `kind`.
An aggregate is **stale** iff any of its input units has a commit newer than
the aggregate's own — read straight off ancestry order, no timestamps, no
bookkeeping:

```python
for index, snapshot in enumerate(repo.ancestry(branch="main")):   # 0 == newest
    ...  # record the smallest index per unit commit and per aggregate commit

stale = newest_input_index[tile] < aggregate_index[tile]   # an input outranks it
```

Reprocessing one year automatically flags its tile's composites for refresh on
the next dispatch — no one has to remember.

## GitHub Actions specifics

**Matrix limit is 256 jobs** → two-level "batches of batches": a dispatch
workflow lists batch indices and pins the snapshot, then a reusable
`workflow_call` workflow expands each batch into its unit matrix (inputs must
be duplicated between `workflow_dispatch` and `workflow_call` blocks — GitHub
quirk):

```yaml
# process_all.yml
jobs:
  generate-batch-matrix:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.gen.outputs.matrix }}
      snapshot: ${{ steps.gen.outputs.snapshot }}
    steps:
      - uses: actions/checkout@v4
      - id: gen
        run: |
          OUT=$(python scripts/get_remaining_work.py --list-batches)   # ancestry fold
          echo "matrix={\"batch\":$(jq -c .batch_index <<<"$OUT")}" >> $GITHUB_OUTPUT
          echo "snapshot=$(jq -r .snapshot_id <<<"$OUT")" >> $GITHUB_OUTPUT

  process-batches:
    needs: generate-batch-matrix
    strategy:
      fail-fast: false                       # one bad unit must not kill the fleet
      matrix: ${{ fromJson(needs.generate-batch-matrix.outputs.matrix) }}
    uses: ./.github/workflows/process_batch.yml   # expands <=256 unit jobs, each:
    with:                                          #   python process_single_unit.py
      batch: ${{ matrix.batch }}                   #     --unit ${{ matrix.unit }}
      as_of_snapshot: ${{ needs.generate-batch-matrix.outputs.snapshot }}
    secrets: inherit                          # storage credentials only
```

- **Timeout + redispatch beats long jobs.** Set a generous `timeout-minutes`
  and let rule 3 absorb whatever doesn't finish; committed sub-units survive.
- **Expect runner roulette.** In one 314-job fleet, ~17% of runners couldn't
  reach an external host on their *first* network call (errno 101 at boot,
  while sibling runners succeeded) — patient, jittered first-contact retries
  (~7 min budget) absorb it. Runner throughput also varies ~3× day to day;
  design unit sizes so the slowest plausible runner still fits inside any
  external deadline you have.
- **Signed-URL lifetimes are a hard wall for big units.** Our imagery URLs
  expire ~45 min after signing: sign *immediately before* each attempt, and
  force a fresh token per attempt — SDK token caches happily hand you one
  with 61 seconds of life left:

```python
planetary_computer.sas.TOKEN_CACHE.clear()    # full-lifetime token per attempt
items = catalog.search(...)                   # signed at search time
```

  If a unit's download still can't fit the window on a median runner, raise
  its I/O parallelism tier or split the unit.

- **Memory**: cap glibc arenas on long multi-step jobs or RSS creeps into
  runner-killing territory, and scale I/O worker counts *per sub-unit* by
  input density — light units get high concurrency for speed; dense units get
  less so peak RSS fits the runner:

```yaml
env:
  MALLOC_ARENA_MAX: "2"     # without this: +3 GB RSS creep over a 10-year job
```

```python
def workers_for(n_inputs):          # I/O is latency-bound: throughput ~ workers,
    if n_inputs <= 150: return 16   # but peak RSS also ~ workers -- densest units
    if n_inputs <= 300: return 12   # must stay inside the 16 GB runner
    return 8
```

## Operating loop, end to end

1. Run the init script once → empty template store.
2. Dispatch the fleet workflow (`which = incomplete`). It folds ancestry,
   pins a snapshot, emits batches.
3. Some jobs fail. Nobody investigates unless a *pattern* emerges (same unit
   failing repeatedly = real bug; scattered one-offs = runner roulette).
4. Redispatch `incomplete` until the fold returns zero. Each pass only runs
   what's missing.
5. Aggregates refresh themselves via staleness. Verification = cross-check a
   sample of store bytes against commit `stats` (they must match exactly),
   plus domain sanity checks.
6. Later reprocessing (algorithm fix, new time slice) is the same loop:
   extend the config/store, dispatch, converge. Priority subsets are just a
   filter on the fold (we processed the ~250 tiles containing validation
   stations first, by passing a tile-list file to the dispatcher).

## Porting checklist

- [ ] Work decomposes into units writing **disjoint regions** of one array store
- [ ] Store template initialized metadata-only; **shard = unit region**; correct fill value
- [ ] Worker entrypoint is platform-agnostic (runs identically on CI, HPC, laptop)
- [ ] One commit per unit; JSON metadata schema versioned from day one
- [ ] Empty-with-reason commits; transient failures never recorded as empty
- [ ] Status fold + snapshot-pinned dispatcher script emitting matrix JSON
- [ ] Bounded commit retry with fresh sessions + conflict rebase
- [ ] Aggregate commits with ancestry-order staleness
- [ ] Loud caps (ceil, floor-of-one) around any "skip broken input" logic
- [ ] Provenance (code SHA, versions, runner id) inside every commit

## When *not* to use this

- Work units can't be made disjoint in the output array (true shared-write
  contention — this pattern has no answer for it).
- Units need more than a runner offers (~16 GB RAM, ~6 h practical wall time
  after retries) even after splitting — though splitting *is* usually the answer.
- You need sub-minute scheduling latency or streaming — CI job startup is ~1 min.
- Private-repo minutes at this scale have a real bill; the economics were
  designed around public-repo runners.

## Scale achieved (for calibration)

~4,400 tiles × 11 years ≈ 47k work units; ~30k Icechunk commits on one
branch; ≈ 5,300 CPU-core-hours and ≈ 60 TB read, entirely on standard GitHub
runners across a handful of dispatch presses over ~1 week; median unit cost
0.9 core-hours; status fold of the full ledger < 1 min; zero data-integrity
incidents attributable to concurrency.
