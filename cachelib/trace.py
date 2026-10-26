"""种子的负载生成器：每个函数都只吃一个 random.Random，因此同种子同参数可完全复现。

trace 是 int 键的列表，值就等于键本身；所有命中率、内存、复用距离的表格都建立
在这些序列之上。真实缓存里"扫描会污染 LRU""老热键会变冷"这两件事，在这里都能
用一条固定序列复现，不需要真实日志。
"""

import math


def uniform(rng, ops, keyspace):
    """均匀独立抽样：LRU 和 FIFO 在这里几乎没有区别，是策略无关的基线。"""
    return [rng.randrange(keyspace) for _ in range(ops)]


def zipf(rng, ops, keyspace, s=0.99):
    """Zipf(s) 抽样：第 k 热的键权重 1/k^s，s 越大热点越尖。"""
    if s <= 0.0:
        raise ValueError("s 必须是正数")
    weights = [1.0 / (k ** s) for k in range(1, keyspace + 1)]
    cdf = []
    total = 0.0
    for w in weights:
        total += w
        cdf.append(total)
    out = []
    for _ in range(ops):
        u = rng.random() * total
        out.append(bisect_index(cdf, u))
    return out


def bisect_index(cdf, u):
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo


def cyclic_scan(rng, ops, hot_size, scan_size, hot_reuses=1):
    """热集 + 周期性整段顺序扫描：LRU 的经典翻车场景。

    每轮先顺序刷完 scan_size 个一次性键，再复用 hot_size 个热键 hot_reuses 次。
    扫描段长度超过容量时，LRU 会把上一轮刚建立的热键全部挤掉。
    """
    out = []
    hot = list(range(hot_size))
    key = hot_size
    while len(out) < ops:
        for _ in range(scan_size):
            out.append(key)
            key += 1
        for _ in range(hot_reuses):
            out.extend(hot)
    return out[:ops]


def time_shift(rng, ops, keyspace, phases, hot_window):
    """分段热区漂移：每段换一个热键窗口，老热键立刻转冷。"""
    if keyspace < hot_window * phases:
        raise ValueError("keyspace 装不下 phases 个不重叠的热区")
    out = []
    step = keyspace // phases
    for p in range(phases):
        start = p * step
        window = list(range(start, start + hot_window))
        count = ops // phases + (1 if p < ops - (ops // phases) * phases else 0)
        for _ in range(count):
            if rng.random() < 0.8:
                out.append(rng.choice(window))
            else:
                out.append(rng.randrange(keyspace))
    return out[:ops]


def powerlaw_reuse(rng, ops, keyspace, alpha=1.0, min_gap=4):
    """栈距离模型：第 pos 次请求指向"往回数 d 个不同键"处的那个键，d 服从 Pareto(alpha)。

    用栈距离而不是"间隔多少条请求"，是因为缓存容量本来就是按不同键数计的：d 直接
    就是任何 LRU 类策略的命中门槛，于是"超过容量的复用份额"这个尾部数字可以精确设定，
    也解释了为什么把缓存加大到某个尺寸之后收益会突然停滞。
    """
    if min_gap < 1:
        raise ValueError("min_gap 至少为 1")

    def sample_gap():
        u = rng.random()
        if u < 1e-12:
            u = 1e-12
        gap = min_gap / (u ** (1.0 / alpha))
        if gap > 1e6:
            gap = 1e6
        return max(min_gap, int(gap))

    stack = []
    out = []
    next_new = 0
    for _ in range(ops):
        gap = sample_gap()
        if gap < len(stack):
            key = stack[gap]
            stack.remove(key)
        elif next_new < keyspace:
            # 目标距离比全部历史还远，只能引入一个新键；栈里全是最多 keyspace 个不同键。
            key = next_new
            next_new += 1
        else:
            # 键空间用尽后把最远的那个键当作"超距复用"，尾部语义保持不变。
            key = stack[-1]
            stack.remove(key)
        stack.insert(0, key)
        out.append(key)
    return out


def reuse_distances(trace):
    """每个复用与其上一次出现之间的距离列表；未命中不计。"""
    last = {}
    out = []
    for pos, key in enumerate(trace):
        p = last.get(key)
        if p is not None:
            out.append(pos - p)
        last[key] = pos
    return out


def summarize(trace):
    """(长度, 不同键数, 复用次数, 距离>容量的复用占比需要外部给容量) 的前三项。"""
    dist = reuse_distances(trace)
    uniq = len(set(trace))
    return {"ops": len(trace), "unique": uniq,
            "reuses": len(dist),
            "mean_gap": (sum(dist) / len(dist)) if dist else 0.0,
            "p50_gap": percentile(dist, 0.5),
            "p95_gap": percentile(dist, 0.95)}


def percentile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(math.ceil(q * len(ordered))) - 1
    if idx < 0:
        idx = 0
    return ordered[idx]


def gap_fraction_above(trace, capacity):
    """复用间隔（两次同名请求之间的操作数）超过 capacity 的比例。

    这个比例就是"把缓存加到 capacity 也仍然救不回来"的复用份额，
    是 powerlaw_reuse 那条表格里唯一能直接讲出口的结论。
    """
    dist = reuse_distances(trace)
    if not dist:
        return 0.0
    over = 0
    for gap in dist:
        if gap > capacity:
            over += 1
    return over / len(dist)
