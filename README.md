# cache-eviction-lru-lfu

A caching engineer picks LRU by habit and then eats a cache-miss spike when a batch job scans
a table once through. This repo measures that failure instead of asserting it: FIFO, LRU,
LRU-with-age-partitioning (2Q-lite) and LFU-with-aging share one packed slot structure (four
parallel arrays plus a key→slot dict, no `OrderedDict`), and every policy is scored against an
exact-optimal MIN/Belady baseline that is handed the future, so heuristic cost is a fraction of
what was achievable. Four findings come out of one run:

1. On the cyclic-scan trace LRU sits at `0.0707` where MIN reaches `0.1400` at the same
   `C=120` — LRU keeps 0.505 of the achievable hit rate, and 2Q-lite closes the whole
   FIFO-to-MIN span (`0.1397`, `clsd 1.000`). The scan-thrash folklore holds exactly.
2. That win needs the scan to overflow the table: at `scan/C=0.25` and `0.50` LRU, 2Q, LFU
   **and** MIN coincide at `0.5693` / `0.3987`, so those rows price no policy choice at all,
   and 2Q is behind LRU in 0 of 5 rows but ahead in only 3.
3. LFU loses where you would expect it to win: on the power-law trace it drops to `0.5360`
   against LRU's `0.9053` and MIN's `0.9448`, while on the time-shift trace the aging knob
   alone is worth `0.4721` (`0.3161` at `decay=1.00`, `0.7883` at `decay=0.00`) — yet even
   that best setting beats LRU by only `+0.0078`.
4. Switching policy is not automatically cheaper than growing the table: on zipf-0.99 at
   `C=300` doubling LRU buys `+0.0972` while the best heuristic switch at fixed memory buys
   `+0.0619`; on the cyclic trace the two levers tie at `+0.0693` each.

Raw output of one run: [`benchmarks/last_run.txt`](benchmarks/last_run.txt). Standard library
only, Python 3.9+. The deterministic columns replay exactly on CPython 3.11; Section 4 counts
`sys.getsizeof` bytes, so that table tracks the interpreter build rather than the workload.

## What it does / does not do

Does:

- Five eviction policies behind one `get`/`put`/`delete` interface, with TTL expiry and
  weighted entries (`cost=n` occupies n slots), all on a hand-written slot store.
- Six seeded traces: uniform, Zipf at s=0.99 and s=1.20, a cyclic sequential scan, a
  time-shifting hot set, a Pareto reuse-distance stream built on LRU stack distance.
- An exact MIN/Belady oracle, plus naive dict/list twins the packed policies are checked
  against step by step, byte accounting per container, timing with a measured noise floor.

Does not:

- It is single-process and in-memory: no external store, no Redis, no sharding, no
  replication, no distributed cache, and `capacity` counts entries/slots rather than bytes.
- MIN is not deployable: it needs the whole future stream, exists only to price the heuristics,
  and its memory row includes the trace index an online policy never pays.
- Timings are wall-clock on a busy machine, so only the `(timing)`-labelled microsecond column
  moves between runs. No concurrency: these caches are not thread-safe and carry no lock.

## Install

Standard library only, Python 3.9+ (verified on CPython 3.11.9); `pip install -e .` registers
the `cachelib` package from `pyproject.toml` if you want it importable elsewhere.

```bash
git clone <this repo> && cd cache-eviction-lru-lfu
python -u -m unittest discover -s tests -p 'test_*.py' -t tests   # 93 tests, OK
```

## Quick start

Every block below was executed on CPython 3.11.9 and the comments are its real output.
The seeds are the benchmark's seeds, so these numbers are the same ones the tables quote.

The thrash case, side by side with the oracle:

```python
import random
from cachelib import LRUCache, TwoQCache, BeladyCache, cyclic_scan, feed

trace = cyclic_scan(random.Random(20261027), 30000, 40, 480, 2)
for cache in (LRUCache(120), TwoQCache(120), BeladyCache(120, trace)):
    feed(cache, trace)
    print('%-12s hit rate %.4f' % (type(cache).__name__, cache.hit_rate))
# LRUCache     hit rate 0.0707
# TwoQCache    hit rate 0.1397
# BeladyCache  hit rate 0.1400
```

LFU is a family, not a policy: the aging knob decides which one you get.

```python
import random
from cachelib import LFUCache, time_shift, feed

trace = time_shift(random.Random(20261028), 30000, 6000, 5, 60)
for decay in (1.0, 0.5, 0.0):
    cache = LFUCache(120, decay=decay, interval=1000)
    feed(cache, trace)
    print('decay %.2f hit rate %.4f sweeps %d' % (decay, cache.hit_rate, cache.sweeps))
# decay 1.00 hit rate 0.3161 sweeps 0
# decay 0.50 hit rate 0.7816 sweeps 30
# decay 0.00 hit rate 0.7883 sweeps 30
```

Weighted entries and TTL are first-class, and an entry that can never fit is refused:

```python
from cachelib import LRUCache

cache = LRUCache(2)
print(cache.put('big', 1, cost=3), cache.rejected)   # False 1
cache.put('a', 1, ttl=10.0, now=0.0)
print(cache.get('a', now=5.0), cache.get('a', now=10.5), len(cache))   # 1 None 0
```

Every policy also has a naive dict/list twin in `cachelib.reference`, and the two agree on the
hit count, the resident set and the last eviction after each single operation (Section 8).

## Measured results

Each block below is copied line-for-line out of `benchmarks/last_run.txt`.

### Hit rate vs cache-size ratio (Section 2, deterministic)

```text
trace zipf-0.99, keyspace 6000, ops 30000
policy             C=30    C=120    C=300    C=600   C=1200
fifo             0.2219   0.3821   0.4989   0.5974   0.6997
lru              0.2621   0.4287   0.5467   0.6439   0.7386
two_q            0.3723   0.5098   0.6086   0.6789   0.7481
lfu              0.4041   0.5003   0.5768   0.6510   0.7386
belady           0.4649   0.6189   0.7175   0.7830   0.8371
```

```text
trace cyclic-scan, keyspace 6000, ops 30000
policy             C=30    C=120    C=300    C=600   C=1200
fifo             0.0000   0.0707   0.0707   0.1053   0.1173
lru              0.0000   0.0707   0.0707   0.1400   0.1400
two_q            0.0000   0.1397   0.1400   0.1400   0.1400
lfu              0.0000   0.1400   0.1400   0.1400   0.1400
belady           0.1033   0.1400   0.1400   0.1400   0.1400
```

On zipf-0.99 `fifo < lru < belady` holds at every size and the two heuristics in the middle
trade places - `lfu 0.4041` beats `two_q 0.3723` at `C=30`, then `two_q` leads from `C=120` on.
One capacity step (`C=300 -> 600`) lifts LRU by `+0.0972` while the best policy switch at
`C=300` is worth `+0.0619`: capacity out-buys policy. On the cyclic trace LRU stays at `0.0707`
from `C=120` to `C=300` while 2Q/LFU already sit at MIN's `0.1400` (switch `+0.0693`, what
doubling buys); at `C=30` the scan exceeds the table and all four heuristics score `0.0000`.

### Distance from exact-optimal MIN per workload (Section 3, deterministic)

```text
trace            fifo     lru      2q     lfu     MIN  lr/MIN    clsd
uniform        0.0207  0.0208  0.0184  0.0204  0.1812   0.115   0.001
zipf-0.99      0.3821  0.4287  0.5098  0.5003  0.6189   0.693   0.539
zipf-1.20      0.6442  0.6944  0.7418  0.7340  0.8091   0.858   0.592
cyclic-scan    0.0707  0.0707  0.1397  0.1400  0.1400   0.505   1.000
time-shift     0.6627  0.7804  0.7892  0.7816  0.8156   0.957   0.828
powerlaw       0.8916  0.9053  0.8924  0.5360  0.9448   0.958   0.258
```

Averaged over the six traces LRU reaches 0.681 of the optimum at `C=120`. Two rows contradict
the easy story. On `uniform` no heuristic passes 0.115 of MIN and 2Q-lite is the worst of the
four (`0.0184`, below FIFO's `0.0207`): the trace has `0.974` of its reuses beyond `C`
(Section 1), so the honest conclusion there is that the cache is the wrong tool, not the wrong
policy. On `powerlaw` LFU-with-aging falls to `0.5360` while FIFO sits at `0.8916` — a heavy
reuse-distance tail keeps counts high on keys that will not return in time.

### Memory per entry per policy (Section 4, deterministic)

```text
policy     struct B    B/slot   rows
fifo         344488     84.10      7
lru          344488     84.10      7
two_q        415808    101.52     10
lfu          412264    100.65     29
belady       738520    180.30      9
```

All five tables are `C=4096` filled by the same zipf-0.99 trace, so FIFO and LRU cost the same
`84.10` B/slot: LRU's recency list is the `store.prev`/`store.nxt` pair FIFO already allocates.
2Q-lite pays `101.52` (1.21x) for `two_q.where` plus the ghost deque and set; LFU pays `100.65`
(1.20x) for a frequency array, a recency array and the live frequency buckets behind its 29
rows; MIN's `180.30` (2.14x) is mostly the index over the trace it was handed.

### Scan pollution: does 2Q-lite actually beat LRU (Section 5, deterministic)

```text
hot set 40 keys, cache C=240, probation budget 60, main budget 180
scan/C        ops      LRU       2Q      LFU      MIN    2Q-LRU   2Q/MIN
0.25        30000   0.5693   0.5693   0.5693   0.5693   +0.0000    1.000
0.50        30000   0.3987   0.3987   0.3987   0.3987   +0.0000    1.000
1.00        30000   0.1240   0.2467   0.2467   0.2467   +0.1227    1.000
2.00        30000   0.0707   0.1400   0.1400   0.1400   +0.0693    1.000
4.00        30000   0.0373   0.0733   0.0441   0.0733   +0.0360    1.000
```

2Q-lite never loses to LRU (the `2Q-LRU` column is never negative; 2 of 5 rows are exact ties)
and its widest win is `+0.1227` at `scan/C=1.00` — double LRU's hit rate for the same 240
slots, at `1.000` of MIN. The `scan/C<=0.5` rows are ties with LRU *and* with MIN, the honest
limit of the claim: while the scan fits inside the table there is no pollution to remove. LFU
equals 2Q at `scan/C=1.00` and `2.00` but falls to `0.0441` against 2Q's `0.0733` at `4.00`.

### LFU aging knob sweep (Section 6, deterministic)

```text
trace time-shift, ops 30000, C=120; cells are hit rate
decay        I=1000    I=10000   I=100000 sweeps@I1
1.00         0.3161     0.3161     0.3161        0
0.90         0.3768     0.3166     0.3161       30
0.70         0.7422     0.3166     0.3161       30
0.50         0.7816     0.3166     0.3161       30
0.20         0.7867     0.3397     0.3161       30
0.00         0.7883     0.7195     0.3161       30
LRU on the same trace: 0.7804    MIN: 0.8156
```

The knob is worth `0.4721` of hit rate on this trace (`0.3161` at `decay=1.00` versus `0.7883`
at `decay=0.00, I=1000`), and the aging *interval* matters as much as the decay: at `I=100000`
every row collapses to `0.3161` because almost no sweep fires, and `decay=1.00` is flat across
intervals with `sweeps 0` since the sweep is skipped. Best LFU beats LRU by `+0.0078` and is
still `0.0273` under MIN.

### Update cost per operation (Section 7, timing)

```text
policy    best us/op  last us/op   spread us    vs FIFO
fifo          1.1173      1.1897      0.0724       1.00
lru           1.3370      1.4858      0.1488       1.20
two_q         1.7887      1.8382      0.0494       1.60
lfu           2.3809      2.5533      0.1724       2.13
belady        1.9059      2.2013      0.2954       1.71
noise floor (largest pass-to-pass spread of an identical batch): 0.2954 us/op
anything inside that is a tie; this machine is running other jobs.
```

Best-of-5 after two discarded warm-ups, `gc` off during sampling, `C=400`, 20000 ops. The
ratios are the conclusion, not the microseconds: LFU is 2.13x FIFO, 2Q-lite 1.60x, LRU 1.20x.
FIFO-vs-LRU (`0.2197` apart) falls inside the `0.2954` floor and is reported as a tie — 2 of
the 10 policy pairs do. LFU's extra hop is the bucket unlink-and-relink per access.

### Correctness (Section 8, deterministic)

```text
policy    over-cap  resurrect  hit-diff  set-diff     steps
fifo             0          0         0         0     21600
lru              0          0         0         0     21600
two_q            0          0         0         0     21600
lfu              0          0         0         0     21600
belady           0          0         0         0     21600
```

`over-cap` counts steps where `len(cache) > capacity`; `resurrect` counts a get returning a key
that had left the cache and was never re-inserted; `hit-diff` and `set-diff` compare against
the naive dict/list reimplementation after **every** operation. 108000 steps over six traces at
capacities 1, 30 and 128, zero of everything. MIN is in the table on purpose: a bug in MIN
would surface as a hit rate above the optimum rather than as a crash, and Section 3 shows
`belady >=` every heuristic on all six rows.

## How it decides

| policy | eviction rule | per-op cost | extra state |
| --- | --- | --- | --- |
| `fifo` | evict the oldest insertion; a read hit changes nothing | O(1) | queue over `prev`/`nxt` |
| `lru` | evict the least recently read or written | O(1) | one list, relinked per hit |
| `two_q` | new keys go to a probation LRU; a second read (or a ghost hit) promotes into the main LRU; probation overflow evicts probation's own tail | O(1) | `where` bytearray + ghost deque/set |
| `lfu` | evict the lowest frequency, ties broken by least recent touch; every `interval` lookups all frequencies are multiplied by `decay` | O(1) amortised, O(capacity) per sweep | freq array, recency array, bucket lists, min-heap |
| `belady` | evict the entry whose next use is farthest away (never used again = first out) | O(log capacity) | forward index over the whole trace |

Two rules that are easy to get wrong are pinned down by tests: a write to a resident key counts
as one access everywhere (it refreshes LRU recency and bumps LFU frequency), and 2Q refuses
promotion when the main budget is 0 (`C=1`), because promoting into a segment that is
immediately trimmed would evict the key you just read. The store behind all five is four
parallel arrays (`keys`, `vals`, `costs`, `exps`) plus a `key→slot` dict, with free slots
threaded through the `exps` array so alloc and release never scan; capacity is weighted, so
`cost=n` occupies n slots and `cost > capacity` is refused outright.

## Limitations

- Mandatory admission: every miss inserts. Real systems reject some misses (TinyLFU/Clock-LFU),
  and Section 5's 2Q result would shift under an admitting filter.
- `capacity` is entries/weighted slots, not bytes: no slab allocator, no per-value measurement,
  and the byte tables only count container `sys.getsizeof`.
- LFU aging is a global multiplicative sweep over resident entries; a production LFU would age a
  window or use decaying counts. The `interval` sweep shows the gap is not cosmetic.
- 2Q-lite keeps no A1in/A1out value semantics beyond key identity and is never retuned:
  `alpha=0.25`, `gamma=0.5` are fixed in every table.
- Traces are synthetic: structurally motivated (scan, drift, Pareto tail) but not a production
  log, and the reuse-distance model is generated rather than measured.
- Single-threaded, no persistence, no `snapshot()`/`restore()`, no statistics windowing.

## API

| name | signature | what it does |
| --- | --- | --- |
| `FIFOCache`, `LRUCache`, `TwoQCache`, `LFUCache`, `BeladyCache` | `(capacity, ...)` | the five policies; `TwoQCache` adds `alpha`, `gamma`; `LFUCache` adds `decay`, `interval`; `BeladyCache` requires the full `trace` |
| `make` | `make(policy, capacity, alpha=, gamma=, decay=, interval=, trace=)` | build a policy by name from `POLICIES` |
| `get`, `put` | `cache.get(key, default=None, now=None)`, `cache.put(key, value, cost=1, ttl=None, now=None)` | read (refreshes recency/frequency, expires TTL) / insert-or-update, evicting until the entry fits; `put` returns `False` if `cost > capacity` |
| `delete`, `purge_expired` | `cache.delete(key)`, `cache.purge_expired(now=None)` | remove one key / drop every expired entry, returning the count |
| `feed` | `feed(cache, trace)` | drive one get (and a put on miss) per trace entry; returns the hit count |
| hit stats | `cache.hits, misses, evictions, rejected, expirations, last_evicted, hit_rate` | counters kept by the shared base class |
| trace generators | `uniform(rng, ops, keyspace)`, `zipf(rng, ops, keyspace, s)`, `cyclic_scan(rng, ops, hot, scan, reuses)`, `time_shift(rng, ops, keyspace, phases, window)`, `powerlaw_reuse(rng, ops, keyspace, alpha, min_gap)` | seeded `list[int]` request streams |
| trace stats | `reuse_distances`, `gap_fraction_above`, `percentile`, `summarize` | distance structure of a stream |
| memory | `detail_rows`, `structure_bytes`, `payload_bytes`, `per_entry_bytes`, `owned_containers`, `int_object_bytes` | `sys.getsizeof` accounting per container |
| reference | `cachelib.reference.RefFIFO`, `RefLRU`, `RefTwoQ`, `RefLFU`, `RefOptimal` | naive dict/list reimplementations for cross-checking |

## Directory tree

```tree
cache-eviction-lru-lfu/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── benchmarks/
│   ├── last_run.txt
│   └── run_benchmarks.py
├── cachelib/
│   ├── __init__.py      # exports policies, trace generators, make(), feed()
│   ├── store.py         # SlotStore + SlotList: the packed arrays and free list
│   ├── base.py          # capacity, TTL, weighted cost, stats, hook protocol
│   ├── fifo.py
│   ├── lru.py
│   ├── two_q.py
│   ├── lfu.py
│   ├── belady.py
│   ├── trace.py
│   ├── memory.py
│   └── reference.py
└── tests/
    └── test_cache.py    # 93 unittest cases
```

## Reproduce

```bash
python -u -m unittest discover -s tests -p 'test_*.py' -t tests
python -u benchmarks/run_benchmarks.py
```

The second command rewrites `benchmarks/last_run.txt` from the same buffer it prints, so the
file is byte-identical to the console output. The deterministic sections replay exactly on
CPython 3.11 (Section 4 adds up `sys.getsizeof` values, so it is build-sensitive); Section 7 is
labelled `(timing)`, prints its own noise floor, and is quoted here as ratios for that reason.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 sparkx313.
