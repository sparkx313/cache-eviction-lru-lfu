"""cache-eviction-lru-lfu：五种淘汰策略 + 一个可复现的命中率基准。

导出分三类：策略（FIFOCache / LRUCache / TwoQCache / LFUCache / BeladyCache）、
trace 生成器与统计（uniform / zipf / cyclic_scan / time_shift / powerlaw_reuse /
reuse_distances / gap_fraction_above / percentile / summarize）、以及驱动与口径
（feed / hit_rate / structure_bytes / per_entry_bytes / reference）。
"""

from .base import Cache, feed, hit_rate
from .belady import BeladyCache
from .fifo import FIFOCache
from .lfu import LFUCache
from .lru import LRUCache
from .memory import (detail_rows, int_object_bytes, owned_containers,
                     payload_bytes, per_entry_bytes, structure_bytes)
from .store import SlotList, SlotStore
from .trace import (cyclic_scan, gap_fraction_above, percentile, powerlaw_reuse,
                    reuse_distances, summarize, time_shift, uniform, zipf)
from .two_q import TwoQCache

__all__ = [
    "Cache", "SlotList", "SlotStore", "POLICIES", "DEFAULT_TRACE_SEED", "make",
    "FIFOCache", "LRUCache", "TwoQCache", "LFUCache", "BeladyCache",
    "feed", "hit_rate",
    "uniform", "zipf", "cyclic_scan", "time_shift", "powerlaw_reuse",
    "reuse_distances", "gap_fraction_above", "percentile", "summarize",
    "detail_rows", "structure_bytes", "payload_bytes", "per_entry_bytes",
    "int_object_bytes", "owned_containers",
]

__version__ = "0.1.0"

POLICIES = ("fifo", "lru", "two_q", "lfu", "belady")

DEFAULT_TRACE_SEED = 20261023


def make(policy, capacity, alpha=0.25, gamma=0.5, decay=0.5, interval=1000,
         trace=None, time_fn=None):
    """按名字构造策略实例，参数全部给默认值，基准与测试都用它取同一组配置。"""
    if policy == "fifo":
        return FIFOCache(capacity, time_fn=time_fn)
    if policy == "lru":
        return LRUCache(capacity, time_fn=time_fn)
    if policy == "two_q":
        return TwoQCache(capacity, alpha=alpha, gamma=gamma, time_fn=time_fn)
    if policy == "lfu":
        return LFUCache(capacity, decay=decay, interval=interval, time_fn=time_fn)
    if policy == "belady":
        if trace is None:
            raise ValueError("belady 需要整条 trace 才能知道未来")
        return BeladyCache(capacity, trace, time_fn=time_fn)
    raise ValueError("未知策略: %r" % (policy,))
