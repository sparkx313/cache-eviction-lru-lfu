"""测量五种缓存淘汰策略在合成但结构明确的负载上到底差多少，并把差距换算成
"离精确最优还差多少"。运行方式：`python -u benchmarks/run_benchmarks.py`。

本文件回答两件事：每个策略在六条 trace 上的命中率与内存占用（确定性列），以及
每次操作的实际开销（计时列）。所有输出都会同时写进同目录的 last_run.txt，README
逐行引用那份文件，所以正文里的数字不可能和这次运行的数据对不上。

确定性列由固定种子的 random.Random 生成，任何机器上逐位复现；计时列不做复现承诺，
每个表都单独标注 (timing)，并给出同批次的实测噪声下限。
"""

import gc
import os
import random
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cachelib import (POLICIES, cyclic_scan, feed, gap_fraction_above, make,
                      powerlaw_reuse, summarize, time_shift, uniform, zipf)
from cachelib import reference as ref
from cachelib.memory import detail_rows, per_entry_bytes, structure_bytes

HEADER = "cache-eviction-lru-lfu benchmarks - python %s" % sys.version.split()[0]
RULE = "=" * 78

OPS = 30000
KEYSPACE = 6000
SEED = 20261023
HOT = 40
SCAN = 480
PHASES = 5
WINDOW = 60
ALPHA = 0.8
MIN_GAP = 6
RATIOS = (0.005, 0.02, 0.05, 0.1, 0.2)
WORK_CAPACITY = int(0.02 * KEYSPACE)
TIME_OPS = 20000
TIME_KEYSPACE = 4000
TIME_CAPACITY = 400
REF_OPS = 1200
REF_CAPACITY = 30
MEMORY_CAPACITY = 4096

REFS = {
    "fifo": lambda cap, trace: ref.RefFIFO(cap),
    "lru": lambda cap, trace: ref.RefLRU(cap),
    "two_q": lambda cap, trace: ref.RefTwoQ(cap),
    "lfu": lambda cap, trace: ref.RefLFU(cap, interval=1000),
    "belady": lambda cap, trace: ref.RefOptimal(cap, trace),
}


def say(text):
    """打印一段解读：统一在 78 列折行，好让 last_run.txt 在 GitHub 上不横向滚动。"""
    print()
    print(textwrap.fill("Interpretation: " + text, 78, subsequent_indent="  "))


def build_traces():
    """六条 trace，一条一个结构问题，全部由固定种子生成。"""
    return [
        ("uniform", uniform(random.Random(SEED + 1), OPS, KEYSPACE)),
        ("zipf-0.99", zipf(random.Random(SEED + 2), OPS, KEYSPACE, 0.99)),
        ("zipf-1.20", zipf(random.Random(SEED + 3), OPS, KEYSPACE, 1.20)),
        ("cyclic-scan", cyclic_scan(random.Random(SEED + 4), OPS, HOT, SCAN, 2)),
        ("time-shift", time_shift(random.Random(SEED + 5), OPS, KEYSPACE,
                                  PHASES, WINDOW)),
        ("powerlaw", powerlaw_reuse(random.Random(SEED + 6), OPS, KEYSPACE,
                                   ALPHA, MIN_GAP)),
    ]


def hit_of(policy, capacity, trace, **kw):
    cache = make(policy, capacity, trace=trace, **kw)
    feed(cache, trace)
    return cache


# ------------------------------------------------------------------- 第 1 节
def section_traces(traces):
    print(RULE)
    print("Section 1. Trace composition (deterministic)")
    print(RULE)
    print("%-13s %8s %8s %8s %8s %8s" %
          ("trace", "ops", "unique", "p50gap", "p95gap", "gap>C"))
    rows = []
    for name, trace in traces:
        info = summarize(trace)
        over = gap_fraction_above(trace, WORK_CAPACITY)
        rows.append((name, info, over))
        print("%-13s %8d %8d %8d %8d %8.3f" %
              (name, info["ops"], info["unique"], info["p50_gap"],
               info["p95_gap"], over))
    widest = max(rows, key=lambda row: row[1]["p95_gap"])
    tightest = min(rows, key=lambda row: row[2])
    say("at C=%d the row that hurts most is %s: p95 reuse distance %d and "
        "%.3f of all reuses land beyond C, which is the part no policy can buy "
        "back by growing the table to this size. %s is the opposite case, with "
        "only %.3f of reuses beyond C."
        % (WORK_CAPACITY, widest[0], widest[1]["p95_gap"], widest[2],
           tightest[0], tightest[2]))
    return rows


# ------------------------------------------------------------------- 第 2 节
def section_size_sweep(traces):
    print(RULE)
    print("Section 2. Hit rate vs cache-size ratio (deterministic)")
    print(RULE)
    caps = [max(1, int(ratio * KEYSPACE)) for ratio in RATIOS]
    chosen = [tr for tr in traces if tr[0] in ("zipf-0.99", "cyclic-scan")]
    grid = {}
    for name, trace in chosen:
        print("trace %s, keyspace %d, ops %d" % (name, KEYSPACE, len(trace)))
        print("%-14s" % "policy" + "".join("%9s" % ("C=%d" % c) for c in caps))
        for policy in POLICIES:
            cells = []
            for cap in caps:
                cache = hit_of(policy, cap, trace)
                cells.append(cache.hit_rate)
                grid[(name, policy, cap)] = cache.hit_rate
            print("%-14s" % policy + "".join("%9.4f" % v for v in cells))
        print()
    mid = caps[2]
    dbl = caps[3]
    for name, _ in chosen:
        best = max(grid[(name, "two_q", mid)], grid[(name, "lfu", mid)])
        best_name = "two_q" if grid[(name, "two_q", mid)] >= grid[(name, "lfu", mid)] else "lfu"
        opt = grid[(name, "belady", mid)]
        lru = grid[(name, "lru", mid)]
        fifo = grid[(name, "fifo", mid)]
        widen = grid[(name, "lru", dbl)] - lru
        swap = best - lru
        print("C=%d -> %d doubles LRU's table on %s" % (mid, dbl, name))
        print("  LRU %.4f -> %.4f (%+.4f)   best policy switch: %.4f (%+.4f)" %
              (lru, grid[(name, "lru", dbl)], widen, best, swap))
        say("%s at C=%d: FIFO %.4f, LRU %.4f, %s %.4f, MIN %.4f. Switching from "
            "LRU to %s is worth %+.4f hit rate at fixed memory, while doubling "
            "LRU's capacity is worth %+.4f; %s the cheaper lever at this size."
            % (name, mid, fifo, lru, best_name, best, opt, best_name, swap, widen,
               "policy is" if abs(swap) >= abs(widen) else "capacity is"))
    return grid


# ------------------------------------------------------------------- 第 3 节
def section_optimal_gap(traces):
    print(RULE)
    print("Section 3. Distance from exact-optimal MIN per workload (deterministic)")
    print(RULE)
    print("%-13s %7s %7s %7s %7s %7s %7s %7s" %
          ("trace", "fifo", "lru", "2q", "lfu", "MIN", "lr/MIN", "clsd"))
    table = []
    for name, trace in traces:
        rates = {}
        for policy in POLICIES:
            rates[policy] = hit_of(policy, WORK_CAPACITY, trace).hit_rate
        span = rates["belady"] - rates["fifo"]
        best = max(rates["lru"], rates["two_q"], rates["lfu"])
        closed = (best - rates["fifo"]) / span if span > 1e-12 else 0.0
        frac = rates["lru"] / rates["belady"] if rates["belady"] else 0.0
        table.append((name, rates, closed, frac))
        print("%-13s %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f %7.3f" %
              (name, rates["fifo"], rates["lru"], rates["two_q"], rates["lfu"],
               rates["belady"], frac, closed))
    print()
    print("MIN is unreachable at runtime; it is reported only to price the others.")
    fracs = [row[3] for row in table]
    worst = min(table, key=lambda row: row[3])
    best = max(table, key=lambda row: row[2])
    say("averaged over the %d traces LRU sits at %.3f of the achievable hit rate "
        "at C=%d. Its worst row is %s at %.3f of optimal (%.4f vs MIN %.4f); the "
        "row where the best heuristic closes most of the FIFO-to-MIN span is %s "
        "(%.3f closed, LRU %.4f, best %.4f, MIN %.4f), so there the policy choice "
        "is nearly the whole story."
        % (len(table), sum(fracs) / float(len(fracs)), WORK_CAPACITY,
           worst[0], worst[3], worst[1]["lru"], worst[1]["belady"], best[0],
           best[2], best[1]["lru"], max(best[1]["lru"], best[1]["two_q"],
                                        best[1]["lfu"]), best[1]["belady"]))
    return table


# ------------------------------------------------------------------- 第 4 节
def section_memory(traces):
    print(RULE)
    print("Section 4. Memory per entry per policy (deterministic)")
    print(RULE)
    index_trace = dict(traces)["zipf-0.99"]
    print("every table is sized at capacity %d and then loaded with the %d-entry" %
          (MEMORY_CAPACITY, len(index_trace)))
    print("zipf-0.99 trace, so the hash map and any per-policy buckets are grown")
    print("%-8s %10s %9s %6s" % ("policy", "struct B", "B/slot", "rows"))
    measured = {}
    fifo_total = None
    for policy in POLICIES:
        cache = make(policy, MEMORY_CAPACITY, trace=index_trace)
        feed(cache, index_trace)
        rows = detail_rows(cache)
        total = structure_bytes(cache)
        if fifo_total is None:
            fifo_total = float(total)
        extras = [label for label, _ in rows if not label.startswith("store.")]
        measured[policy] = (total, per_entry_bytes(cache), len(rows), extras)
        print("%-8s %10d %9.2f %6d" %
              (policy, total, per_entry_bytes(cache), len(rows)))
        print("         extras: " +
              textwrap.fill(", ".join(extras) if extras else "-", 58,
                            subsequent_indent="         extras: ").lstrip())
    lfu_ratio = measured["lfu"][0] / fifo_total
    two_ratio = measured["two_q"][0] / fifo_total
    min_ratio = measured["belady"][0] / fifo_total
    lru_ratio = measured["lru"][0] / fifo_total
    say("bytes per slot with the table full: FIFO %.2f, LRU %.2f (%.2fx), 2Q %.2f "
        "(%.2fx), LFU %.2f (%.2fx), MIN %.2f (%.2fx). 2Q's excess is the segment "
        "bytearray plus %d ghost containers; LFU's is the frequency array, the "
        "recency array and %d live frequency buckets on top of the shared %d "
        "store arrays. MIN is %.2fx because it keeps an index over the %d-entry "
        "trace it was handed, which no online policy has to carry."
        % (measured["fifo"][1], measured["lru"][1], lru_ratio, measured["two_q"][1],
           two_ratio, measured["lfu"][1], lfu_ratio, measured["belady"][1],
           min_ratio, len(measured["two_q"][3]), len(measured["lfu"][3]) - 4,
           measured["fifo"][2], min_ratio, len(index_trace)))
    return measured


# ------------------------------------------------------------------- 第 5 节
def section_scan_pollution():
    print(RULE)
    print("Section 5. Scan pollution: 2Q-lite vs LRU on a cyclic scan (deterministic)")
    print(RULE)
    cap = 240
    probe = make("two_q", cap)
    print("hot set %d keys, cache C=%d, probation budget %d, main budget %d" %
          (HOT, cap, probe.prob_budget, probe.main_budget))
    print("%-9s %7s %8s %8s %8s %8s %9s %8s" %
          ("scan/C", "ops", "LRU", "2Q", "LFU", "MIN", "2Q-LRU", "2Q/MIN"))
    rows = []
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        scan = max(8, int(factor * cap))
        trace = cyclic_scan(random.Random(SEED + 40), OPS, HOT, scan, 2)
        rates = {}
        for policy in ("lru", "two_q", "lfu", "belady"):
            rates[policy] = hit_of(policy, cap, trace).hit_rate
        rows.append((factor, rates, len(trace), scan))
        print("%-9.2f %7d %8.4f %8.4f %8.4f %8.4f %+9.4f %8.3f" %
              (factor, len(trace), rates["lru"], rates["two_q"], rates["lfu"],
               rates["belady"], rates["two_q"] - rates["lru"],
               (rates["two_q"] / rates["belady"] if rates["belady"] else 0.0)))
    widest = max(rows, key=lambda row: row[1]["two_q"] - row[1]["lru"])
    losses = [row for row in rows if row[1]["two_q"] < row[1]["lru"]]
    ties = [row for row in rows if abs(row[1]["two_q"] - row[1]["lru"]) < 1e-12]
    print()
    print("rows where 2Q is behind LRU: %d of %d; identical: %d" %
          (len(losses), len(rows), len(ties)))
    verdict = ("2Q-lite never falls below LRU on this family"
               if not losses else
               "2Q-lite falls below LRU at scan/C=%s" %
               ",".join("%.2f" % row[0] for row in losses))
    say("at its widest row, scan/C=%.2f with scan=%d keys per cycle, LRU is at "
        "%.4f and 2Q-lite at %.4f, i.e. %+.4f hit rate for the same %d slots, and "
        "%.3f of MIN's %.4f. %s: at the smallest row (scan/C=%.2f) LRU %.4f, 2Q "
        "%.4f and MIN %.4f coincide, so that row prices no policy choice at all."
        % (widest[0], widest[3], widest[1]["lru"], widest[1]["two_q"],
           widest[1]["two_q"] - widest[1]["lru"], cap,
           (widest[1]["two_q"] / widest[1]["belady"] if widest[1]["belady"] else 0.0),
           widest[1]["belady"], verdict, rows[0][0], rows[0][1]["lru"],
           rows[0][1]["two_q"], rows[0][1]["belady"]))
    return rows


# ------------------------------------------------------------------- 第 6 节
def section_aging(traces):
    print(RULE)
    print("Section 6. LFU aging knob sweep (deterministic)")
    print(RULE)
    trace = dict(traces)["time-shift"]
    decays = (1.0, 0.9, 0.7, 0.5, 0.2, 0.0)
    intervals = (1000, 10000, 100000)
    print("trace time-shift, ops %d, C=%d; cells are hit rate" %
          (len(trace), WORK_CAPACITY))
    print("%-8s" % "decay" + "".join("%11s" % ("I=%d" % i) for i in intervals)
          + "%10s" % "sweeps@I1")
    grid = {}
    for decay in decays:
        cells = []
        for interval in intervals:
            cache = hit_of("lfu", WORK_CAPACITY, trace, decay=decay,
                           interval=interval)
            grid[(decay, interval)] = (cache.hit_rate, cache.sweeps)
            cells.append(cache.hit_rate)
        print("%-8.2f" % decay + "".join("%11.4f" % v for v in cells)
              + "%9d" % grid[(decay, intervals[0])][1])
    static = hit_of("lru", WORK_CAPACITY, trace).hit_rate
    opt = hit_of("belady", WORK_CAPACITY, trace).hit_rate
    print("LRU on the same trace: %.4f    MIN: %.4f" % (static, opt))
    best = max(grid, key=lambda key: grid[key][0])
    worst = min(grid, key=lambda key: grid[key][0])
    no_aging = grid[(1.0, intervals[2])][0]
    flat = [grid[(1.0, i)][0] for i in intervals]
    tied = sum(1 for i in intervals
               if abs(grid[(worst[0], i)][0] - grid[worst][0]) < 1e-12)
    print("decay=1.00 is flat across intervals (%s), sweeps=%d:" %
          (", ".join("%.4f" % v for v in flat), grid[(1.0, intervals[0])][1]))
    print("with no decay the sweep is skipped: that row has no knob left")
    say("the knob alone moves LFU by %.4f hit rate: best is decay %.2f at "
        "interval %d (%.4f, %d sweeps), worst is decay %.2f at interval %d "
        "(%.4f, %d sweeps, %d interval cells tie at that rate). Against LRU at "
        "%.4f the best setting gains %+.4f and the worst loses %+.4f, and MIN "
        "sits at %.4f, so \"LFU\" here is a family whose endpoints differ more "
        "than the distance from LRU to its own best alternative. The flat row is "
        "the reason: leaving aging off entirely sits %.4f below LRU on this trace."
        % (grid[best][0] - grid[worst][0], best[0], best[1], grid[best][0],
           grid[best][1], worst[0], worst[1], grid[worst][0], grid[worst][1],
           tied, static, grid[best][0] - static, grid[worst][0] - static, opt,
           static - no_aging))
    return grid


# ------------------------------------------------------------------- 第 7 节
def bench_ops(policy, trace, capacity, repeats=5, warmups=2):
    """同一批次重复若干遍，返回 (最佳每操作秒数, 逐遍结果)。采样期间关掉 gc。"""
    passes = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(warmups):
            cache = make(policy, capacity, trace=trace)
            feed(cache, trace)
        for _ in range(repeats):
            cache = make(policy, capacity, trace=trace)
            cache.clear_stats()
            start = time.perf_counter()
            feed(cache, trace)
            passes.append((time.perf_counter() - start) / float(cache.lookups))
    finally:
        gc.enable()
    return min(passes), passes


def section_update_cost():
    print(RULE)
    print("Section 7. Update cost per operation (timing)")
    print(RULE)
    trace = zipf(random.Random(SEED + 70), TIME_OPS, TIME_KEYSPACE, 0.99)
    floor = 0.0
    results = {}
    print("ops %d, C=%d, 2 warm-ups discarded then best-of-5." %
          (len(trace), TIME_CAPACITY))
    print("Construction, including MIN's trace index, is outside the timed region.")
    print("%-8s %11s %11s %11s %10s" %
          ("policy", "best us/op", "last us/op", "spread us", "vs FIFO"))
    for policy in POLICIES:
        best_s, passes = bench_ops(policy, trace, TIME_CAPACITY)
        spread = (max(passes) - min(passes)) * 1e6
        floor = max(floor, spread)
        results[policy] = (best_s * 1e6, spread, max(passes) * 1e6)
    fifo = results["fifo"][0]
    for policy in POLICIES:
        print("%-8s %11.4f %11.4f %11.4f %10.2f" %
              (policy, results[policy][0], results[policy][2],
               results[policy][1], results[policy][0] / fifo))
    print("noise floor (largest pass-to-pass spread of an identical batch): "
          "%.4f us/op" % floor)
    print("anything inside that is a tie; this machine is running other jobs.")
    separated = [p for p in POLICIES
                 if p != "fifo" and abs(results[p][0] - fifo) > floor]
    slowest = max(POLICIES, key=lambda p: results[p][0])
    cheapest = min(POLICIES, key=lambda p: results[p][0])
    pairs = []
    for i, a in enumerate(POLICIES):
        for b in POLICIES[i + 1:]:
            diff = abs(results[a][0] - results[b][0])
            pairs.append((diff, a, b))
    pairs.sort(reverse=True)
    indistinguishable = [pair for pair in pairs if pair[0] <= floor]
    say("best-of-5 per operation: %s %.4f us/op, %s %.4f us/op, ratio %.2f. "
        "%d of the %d non-FIFO policies separate from FIFO by more than the "
        "%.4f us/op floor, and %d of the %d policy pairs are inside the floor and "
        "therefore ties. LFU's extra %.4f us/op over FIFO is the cost of the "
        "bucket bookkeeping, not of a different asymptotic class."
        % (slowest, results[slowest][0], cheapest, results[cheapest][0],
           results[slowest][0] / results[cheapest][0], len(separated),
           len(POLICIES) - 1, floor, len(indistinguishable), len(pairs),
           results["lfu"][0] - fifo))
    return results


# ------------------------------------------------------------------- 第 8 节
def section_correctness(traces):
    print(RULE)
    print("Section 8. Correctness vs naive reference (deterministic)")
    print(RULE)
    short = [(name, trace[:REF_OPS]) for name, trace in traces]
    caps = (1, REF_CAPACITY, 128)
    counts = {}
    print("%-8s %9s %10s %9s %9s %9s" %
          ("policy", "over-cap", "resurrect", "hit-diff", "set-diff", "steps"))
    for policy in POLICIES:
        over_cap = resurrect = disagree = mismatch = steps = 0
        for name, trace in short:
            for cap in caps:
                cache = make(policy, cap, trace=trace)
                shadow = REFS[policy](cap, trace)
                gone = set()
                for key in trace:
                    steps += 1
                    if len(cache) > cap:
                        over_cap += 1
                    cache.last_evicted = None
                    value = cache.get(key)
                    if cache.last_evicted is not None:
                        gone.add(cache.last_evicted)
                    if value is None:
                        cache.put(key, key)
                        if cache.last_evicted is not None:
                            gone.add(cache.last_evicted)
                        gone.discard(key)
                    elif key in gone:
                        resurrect += 1
                    if shadow.get(key) is None:
                        shadow.put(key, key)
                    if set(cache) != shadow.residents():
                        mismatch += 1
                    if cache.hits != shadow.hits:
                        disagree += 1
        counts[policy] = (over_cap, resurrect, disagree, mismatch, steps)
        print("%-8s %9d %10d %9d %9d %9d" %
              (policy, over_cap, resurrect, disagree, mismatch, steps))
    total = sum(counts[p][4] for p in POLICIES)
    sums = [sum(counts[p][i] for p in POLICIES) for i in (0, 1, 2, 3)]
    bad = sum(sums)
    say("%d (trace, capacity) steps over %d traces at capacities %s, each policy "
        "watched against its own naive dict/list reimplementation: %d capacity "
        "violations, %d returns of an already-evicted key, %d hit-count "
        "disagreements, %d resident-set mismatches, %d defects in total. MIN is "
        "included in that count, which is what makes it the calibration row: a "
        "bug in MIN would show up as a hit rate above the optimum, not as a crash."
        % (total, len(short), caps, sums[0], sums[1], sums[2], sums[3], bad))
    return counts


class _Tee:
    """照常打印，同时把全部内容缓存下来写进文件。

    这样 last_run.txt 就是这一次运行的逐字记录，README 里的数字不可能和它不一致。
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.path = path
        self.parts = []

    def write(self, text):
        self.stream.write(text)
        self.parts.append(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def save(self):
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(self.parts))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run.txt")
    console = sys.stdout
    sys.stdout = _Tee(console, report)
    try:
        print(HEADER)
        print("This file measures hit rate, memory, update cost and correctness of")
        print("five eviction policies against the exact-optimal MIN baseline.")
        print()
        traces = build_traces()
        section_traces(traces)
        print()
        section_size_sweep(traces)
        print()
        section_optimal_gap(traces)
        print()
        section_memory(traces)
        print()
        section_scan_pollution()
        print()
        section_aging(traces)
        print()
        section_update_cost()
        print()
        section_correctness(traces)
    finally:
        sys.stdout.save()
        sys.stdout = console
        print("wrote %s" % report)


if __name__ == "__main__":
    main()
