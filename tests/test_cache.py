"""cachelib 的不变量与边界测试：容量、淘汰顺序、老化、TTL、加权成本、参考实现一致性。

每个策略都跑三件事：手写的小序列（验证淘汰顺序本身）、同一条种子 trace 与朴素参考
实现逐步比对驻留集合、以及容量/复活不变量的随机 trace 压力测试。
"""

import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cachelib import (POLICIES, SlotList, SlotStore, TwoQCache, cyclic_scan,
                     feed, gap_fraction_above, make, percentile, powerlaw_reuse,
                     reuse_distances, summarize, time_shift, uniform, zipf)
from cachelib import reference as ref
from cachelib.fifo import FIFOCache
from cachelib.lfu import LFUCache
from cachelib.lru import LRUCache
from cachelib.memory import (detail_rows, int_object_bytes, owned_containers,
                             payload_bytes, per_entry_bytes, structure_bytes)

INTERVAL = 137

REFS = {
    "fifo": lambda cap, trace: ref.RefFIFO(cap),
    "lru": lambda cap, trace: ref.RefLRU(cap),
    "two_q": lambda cap, trace: ref.RefTwoQ(cap),
    "lfu": lambda cap, trace: ref.RefLFU(cap, interval=INTERVAL),
    "belady": lambda cap, trace: ref.RefOptimal(cap, trace),
}


def build(policy, capacity, trace=None, **kw):
    kw.setdefault("interval", INTERVAL)
    return make(policy, capacity, trace=trace, **kw)


def sample_traces(n=1500):
    return {
        "uniform": uniform(random.Random(11), n, 240),
        "zipf": zipf(random.Random(12), n, 240, 0.99),
        "scan": cyclic_scan(random.Random(13), n, 12, 90, 2),
        "shift": time_shift(random.Random(14), n, 240, 3, 20),
        "heavy": powerlaw_reuse(random.Random(15), n, 400, 0.8, 6),
    }


# --------------------------------------------------------------------------- 结构层
class TestSlotList(unittest.TestCase):
    def setUp(self):
        self.prev = [-1] * 5
        self.nxt = [-1] * 5
        self.lst = SlotList(self.prev, self.nxt)

    def test_push_front_order(self):
        for i in (0, 1, 2):
            self.lst.push_front(i)
        self.assertEqual(self.lst.indexes(), [2, 1, 0])

    def test_push_back_order(self):
        for i in (0, 1, 2):
            self.lst.push_back(i)
        self.assertEqual(self.lst.indexes(), [0, 1, 2])

    def test_unlink_middle_and_ends(self):
        for i in (0, 1, 2):
            self.lst.push_back(i)
        self.lst.unlink(1)
        self.assertEqual(self.lst.indexes(), [0, 2])
        self.lst.unlink(0)
        self.lst.unlink(2)
        self.assertEqual(self.lst.indexes(), [])
        self.assertEqual(len(self.lst), 0)

    def test_move_to_front(self):
        for i in (0, 1, 2):
            self.lst.push_back(i)
        self.lst.move_to_front(0)
        self.assertEqual(self.lst.indexes(), [0, 1, 2])
        self.lst.move_to_front(2)
        self.assertEqual(self.lst.indexes(), [2, 0, 1])

    def test_pop_ends(self):
        for i in (0, 1, 2):
            self.lst.push_back(i)
        self.assertEqual(self.lst.pop_front(), 0)
        self.assertEqual(self.lst.pop_back(), 2)
        self.assertEqual(self.lst.indexes(), [1])

    def test_pop_on_empty_returns_minus_one(self):
        self.assertEqual(self.lst.pop_front(), -1)
        self.assertEqual(self.lst.pop_back(), -1)


class TestSlotStore(unittest.TestCase):
    def test_alloc_release_reuses_slots(self):
        s = SlotStore(3)
        a = s.alloc("a", 1)
        b = s.alloc("b", 2)
        self.assertEqual(s.used, 2)
        s.release(a)
        c = s.alloc("c", 3)
        self.assertEqual(c, a)
        self.assertEqual(len(s), 2)
        self.assertEqual(s.free_count, 1)

    def test_free_chain_wraps_all_slots(self):
        s = SlotStore(2)
        x = s.alloc("x", 1)
        y = s.alloc("y", 2)
        s.release(x)
        s.release(y)
        self.assertEqual(s.free_head, y)
        self.assertEqual(s.exps[y], x)
        self.assertEqual(s.exps[x], -1)
        self.assertEqual(s.alloc("z", 3), y)

    def test_used_units_with_cost(self):
        s = SlotStore(5)
        i = s.alloc("k", 1, 3)
        self.assertEqual(s.used, 3)
        s.set_cost(i, 5)
        self.assertEqual(s.used, 5)
        s.release(i)
        self.assertEqual(s.used, 0)

    def test_make_list_is_lazy_and_shared(self):
        s = SlotStore(4)
        self.assertIsNone(s.prev)
        one = s.make_list()
        two = s.make_list()
        self.assertIs(one.prev, two.prev)
        self.assertIs(one.nxt, two.nxt)

    def test_alloc_on_full_pool_raises(self):
        s = SlotStore(1)
        s.alloc("k", 1)
        self.assertRaises(RuntimeError, s.alloc, "j", 2)

    def test_detail_rows_cover_every_array(self):
        s = SlotStore(4)
        labels = [name for name, _ in s.detail()]
        self.assertIn("store.keys", labels)
        self.assertNotIn("store.prev", labels)
        s.make_list()
        self.assertIn("store.nxt", [name for name, _ in s.detail()])


# --------------------------------------------------------------------------- FIFO
class TestFifo(unittest.TestCase):
    def setUp(self):
        self.c = FIFOCache(3)

    def test_evicts_in_insertion_order(self):
        for key in ("a", "b", "c"):
            self.c.put(key, key)
        self.c.put("d", "d")
        self.assertEqual(self.c.resident_order(), ["b", "c", "d"])
        self.assertEqual(self.c.last_evicted, "a")

    def test_get_does_not_reorder(self):
        for key in ("a", "b", "c"):
            self.c.put(key, key)
        self.c.get("a")
        self.c.put("d", "d")
        self.assertEqual(self.c.resident_order(), ["b", "c", "d"])

    def test_duplicate_put_updates_value_only(self):
        self.c.put("a", 1)
        self.c.put("a", 2)
        self.assertEqual(len(self.c), 1)
        self.assertEqual(self.c.get("a"), 2)

    def test_missing_key_returns_default(self):
        self.assertIsNone(self.c.get("nope"))
        self.assertEqual(self.c.get("nope", "dflt"), "dflt")
        self.assertEqual(self.c.misses, 2)

    def test_size_one(self):
        one = FIFOCache(1)
        one.put("a", 1)
        one.put("b", 2)
        self.assertEqual(list(one), ["b"])
        self.assertEqual(len(one), 1)

    def test_zero_capacity_stores_nothing(self):
        zero = FIFOCache(0)
        zero.put("a", 1)
        self.assertEqual(len(zero), 0)
        self.assertIsNone(zero.get("a"))
        self.assertFalse(zero.put("b", 2))

    def test_capacity_never_exceeded_on_trace(self):
        trace = zipf(random.Random(21), 600, 90, 0.99)
        feed(self.c, trace)
        self.assertLessEqual(len(self.c), 3)


# --------------------------------------------------------------------------- LRU
class TestLru(unittest.TestCase):
    def setUp(self):
        self.c = LRUCache(3)

    def test_hit_moves_to_front(self):
        for key in ("a", "b", "c"):
            self.c.put(key, key)
        self.c.get("a")
        self.c.put("d", "d")
        self.assertEqual(self.c.resident_order(), ["d", "a", "c"])
        self.assertEqual(self.c.last_evicted, "b")

    def test_write_updates_recency(self):
        for key in ("a", "b", "c"):
            self.c.put(key, key)
        self.c.put("a", "A")
        self.c.put("d", "d")
        self.assertEqual(self.c.get("a"), "A")
        self.assertEqual(self.c.last_evicted, "b")

    def test_eviction_order_after_mixed_access(self):
        for key in "abc":
            self.c.put(key, key)
        self.c.get("c")
        self.c.get("b")
        self.c.put("d", "d")
        self.assertEqual(self.c.last_evicted, "a")

    def test_size_one_and_zero(self):
        one = LRUCache(1)
        one.put("a", 1)
        one.get("a")
        one.put("b", 2)
        self.assertEqual(set(one), {"b"})
        self.assertEqual(len(LRUCache(0)), 0)

    def test_missing_key_counts_miss(self):
        self.c.get("ghost")
        self.assertEqual(self.c.misses, 1)
        self.assertEqual(self.c.hits, 0)
        self.assertEqual(self.c.hit_rate, 0.0)

    def test_delete_frees_slot(self):
        self.c.put("a", 1)
        self.assertTrue(self.c.delete("a"))
        self.assertFalse(self.c.delete("a"))
        self.assertEqual(len(self.c), 0)
        self.assertEqual(self.c.last_evicted, "a")

    def test_iter_and_contains(self):
        self.c.put("a", 1)
        self.assertIn("a", self.c)
        self.assertNotIn("z", self.c)
        self.assertEqual(list(self.c), ["a"])


# --------------------------------------------------------------------------- 2Q-lite
class TestTwoQ(unittest.TestCase):
    def test_first_touch_lands_in_probation(self):
        c = TwoQCache(8)
        c.put("a", "a")
        self.assertEqual(c.probation_length, 1)
        self.assertEqual(c.main_length, 0)

    def test_second_read_promotes_to_main(self):
        c = TwoQCache(8)
        c.put("a", "a")
        c.put("b", "b")
        c.get("a")
        self.assertEqual(c.main_length, 1)
        self.assertEqual(c.probation_length, 1)
        self.assertEqual(c.promotions, 1)

    def test_ghost_reinsert_enters_main(self):
        c = TwoQCache(8)
        c.put("a", "a")
        c.put("b", "b")
        c.put("c", "c")
        self.assertEqual(c.last_evicted, "a")
        self.assertIn("a", c.ghost_set)
        self.assertIsNone(c.get("a"))
        c.put("a", "a")
        self.assertEqual(c.main_length, 1)
        self.assertEqual(c.ghost_hits, 1)

    def test_budgets_split_capacity(self):
        c = TwoQCache(64)
        self.assertEqual(c.prob_budget, 16)
        self.assertEqual(c.main_budget, 48)
        small = TwoQCache(3)
        self.assertEqual((small.prob_budget, small.main_budget), (1, 2))

    def test_segments_sum_to_length(self):
        c = TwoQCache(12)
        feed(c, zipf(random.Random(22), 500, 60, 1.0))
        self.assertEqual(c.probation_length + c.main_length, len(c))
        self.assertLessEqual(len(c), 12)

    def test_scan_does_not_flush_hot_set(self):
        c = TwoQCache(16)
        hot = list(range(6))
        for _ in range(2):
            for key in hot:
                if c.get(key) is None:
                    c.put(key, key)
        for key in range(1000, 1040):
            c.put(key, key)
        self.assertEqual(c.main_length, 6)
        for key in hot:
            self.assertEqual(c.get(key), key)

    def test_capacity_one_never_evicts_itself(self):
        c = TwoQCache(1)
        c.put("a", 1)
        for _ in range(4):
            c.get("a")
        self.assertEqual(len(c), 1)
        self.assertEqual(c.get("a"), 1)

    def test_bad_alpha_rejected(self):
        self.assertRaises(ValueError, TwoQCache, 8, 0.0)
        self.assertRaises(ValueError, TwoQCache, 8, 1.0)

    def test_ghost_budget_bounds_records(self):
        c = TwoQCache(8, gamma=0.5)
        for key in range(200):
            c.put(key, key)
        self.assertLessEqual(c.ghost_length, c.ghost_budget)


# --------------------------------------------------------------------------- LFU
class TestLfu(unittest.TestCase):
    def test_evicts_lowest_freq(self):
        c = LFUCache(3, decay=1.0, interval=10 ** 6)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        for _ in range(3):
            c.get("a")
            c.get("b")
        c.put("d", 4)
        self.assertEqual(c.last_evicted, "c")

    def test_tie_breaks_on_recency(self):
        c = LFUCache(2, decay=1.0, interval=10 ** 6)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.get("b")
        c.put("c", 3)
        self.assertEqual(c.last_evicted, "a")

    def test_no_aging_when_decay_is_one(self):
        c = LFUCache(4, decay=1.0, interval=3)
        feed(c, zipf(random.Random(23), 200, 30, 0.99))
        self.assertEqual(c.sweeps, 0)

    def test_sweep_fires_at_interval(self):
        c = LFUCache(4, decay=0.5, interval=5)
        c.put("a", 1)
        for _ in range(10):
            c.get("zzz-missing")
        self.assertEqual(c.sweeps, 2)

    def test_sweep_multiplies_freq(self):
        c = LFUCache(8, decay=0.5, interval=10)
        for key in "abcdefgh":
            c.put(key, key)
        for _ in range(4):
            c.get("a")
        for _ in range(5):
            c.get("b")
        c.get("a")
        self.assertEqual(c.frequencies()["a"], 3)
        self.assertEqual(c.frequencies()["b"], 3)

    def test_decay_zero_flattens_every_frequency(self):
        c = LFUCache(4, decay=0.0, interval=2)
        c.put("a", 1)
        c.put("b", 2)
        for _ in range(6):
            c.get("a")
        self.assertEqual(set(c.frequencies().values()), {0, 1})

    def test_write_counts_as_access(self):
        c = LFUCache(4, decay=1.0, interval=10 ** 6)
        c.put("a", 1)
        c.put("a", 2)
        c.put("a", 3)
        self.assertEqual(c.frequencies()["a"], 3)

    def test_capacity_one(self):
        c = LFUCache(1)
        trace = zipf(random.Random(24), 200, 30, 0.99)
        r = ref.RefLFU(1, interval=INTERVAL)
        feed(c, trace)
        feed(r, trace)
        self.assertLessEqual(len(c), 1)
        self.assertEqual(c.hits, r.hits)

    def test_victim_always_found_under_churn(self):
        c = LFUCache(6, decay=0.5, interval=13)
        feed(c, zipf(random.Random(25), 900, 50, 1.2))
        self.assertEqual(len(c), 6)

    def test_bad_params_rejected(self):
        self.assertRaises(ValueError, LFUCache, 4, 1.5, 10)
        self.assertRaises(ValueError, LFUCache, 4, 0.5, 0)

    def test_hit_rate_matches_reference(self):
        trace = zipf(random.Random(26), 400, 60, 0.99)
        c = LFUCache(20, decay=0.5, interval=INTERVAL)
        r = ref.RefLFU(20, interval=INTERVAL)
        feed(c, trace)
        feed(r, trace)
        self.assertEqual(c.hits, r.hits)


# --------------------------------------------------------------------------- MIN
class TestBelady(unittest.TestCase):
    def test_matches_reference_optimal_hits(self):
        trace = zipf(random.Random(27), 500, 80, 0.99)
        c = build("belady", 20, trace)
        r = ref.RefOptimal(20, trace)
        feed(c, trace)
        feed(r, trace)
        self.assertEqual(c.hits, r.hits)

    def test_hits_are_an_upper_bound(self):
        trace = zipf(random.Random(28), 500, 80, 0.99)
        best = build("belady", 20, trace)
        feed(best, trace)
        for policy in ("fifo", "lru", "two_q", "lfu"):
            other = build(policy, 20, trace)
            feed(other, trace)
            self.assertLessEqual(other.hits, best.hits)

    def test_capacity_zero_and_full(self):
        trace = uniform(random.Random(29), 300, 40)
        zero = build("belady", 0, trace)
        feed(zero, trace)
        self.assertEqual((len(zero), zero.hits), (0, 0))
        big = build("belady", 40, trace)
        feed(big, trace)
        self.assertEqual(big.misses, len(set(trace)))

    def test_cursor_tracks_feed(self):
        trace = uniform(random.Random(30), 120, 20)
        c = build("belady", 5, trace)
        feed(c, trace)
        self.assertEqual(c.cursor, len(trace) - 1)

    def test_missing_trace_rejected(self):
        self.assertRaises(ValueError, make, "belady", 4)

    def test_residency_matches_reference_step_by_step(self):
        trace = zipf(random.Random(31), 400, 60, 1.1)
        c = build("belady", 12, trace)
        r = ref.RefOptimal(12, trace)
        for key in trace:
            if c.get(key) is None:
                c.put(key, key)
            if r.get(key) is None:
                r.put(key, key)
            self.assertEqual(set(c), r.residents())

    def test_never_used_keys_leave_first(self):
        trace = [1, 2, 3, 4, 5, 1, 2, 3]
        c = build("belady", 3, trace)
        feed(c, trace)
        self.assertLessEqual(set(c), {1, 2, 3})


# --------------------------------------------------------------------------- TTL
class TestTtl(unittest.TestCase):
    def setUp(self):
        self.c = LRUCache(3)

    def test_expiry_on_get(self):
        self.c.put("a", 1, ttl=10.0, now=100.0)
        self.assertEqual(self.c.get("a", now=105.0), 1)
        self.assertIsNone(self.c.get("a", now=111.0))
        self.assertEqual(len(self.c), 0)

    def test_expired_read_counts_as_miss(self):
        self.c.put("a", 1, ttl=1.0, now=0.0)
        self.c.get("a", now=2.0)
        self.assertEqual((self.c.hits, self.c.misses, self.c.expirations), (0, 1, 1))

    def test_put_refreshes_ttl(self):
        self.c.put("a", 1, ttl=5.0, now=0.0)
        self.c.put("a", 2, ttl=5.0, now=4.0)
        self.assertEqual(self.c.get("a", now=8.0), 2)
        self.assertIsNone(self.c.get("a", now=10.0))

    def test_purge_expired(self):
        self.c.put("a", 1, ttl=1.0, now=0.0)
        self.c.put("b", 2, ttl=100.0, now=0.0)
        self.c.put("c", 3)
        self.assertEqual(self.c.purge_expired(now=2.0), 1)
        self.assertEqual(len(self.c), 2)

    def test_contains_respects_expiry(self):
        clock = [100.0]
        c = LRUCache(3, time_fn=lambda: clock[0])
        c.put("a", 1, ttl=50.0, now=100.0)
        c.put("b", 2, ttl=5.0, now=100.0)
        self.assertIn("a", c)
        clock[0] = 120.0
        self.assertNotIn("b", c)
        self.assertIn("a", c)

    def test_expired_slot_is_reusable(self):
        self.c.put("a", 1, ttl=1.0, now=0.0)
        self.c.get("a", now=5.0)
        self.c.put("b", 2)
        self.assertEqual(len(self.c), 1)
        self.assertEqual(list(self.c), ["b"])

    def test_put_on_expired_key_reinserts(self):
        self.c.put("a", 1, ttl=1.0, now=0.0)
        self.c.put("a", 9, now=5.0)
        self.assertEqual(self.c.get("a"), 9)
        self.assertEqual(len(self.c), 1)

    def test_wall_clock_expiry(self):
        c = LRUCache(2)
        c.put("a", 1, ttl=0.01)
        time.sleep(0.03)
        self.assertIsNone(c.get("a"))


# --------------------------------------------------------------------------- 加权成本
class TestWeightedCost(unittest.TestCase):
    def test_cost_above_capacity_rejected(self):
        c = LRUCache(4)
        self.assertFalse(c.put("big", 1, cost=5))
        self.assertEqual(len(c), 0)
        self.assertEqual(c.rejected, 1)
        self.assertIsNone(c.get("big"))

    def test_weighted_units_evict_until_fit(self):
        c = FIFOCache(5)
        c.put("a", 1, cost=3)
        c.put("b", 2, cost=3)
        self.assertEqual(len(c), 1)
        self.assertEqual(list(c), ["b"])
        self.assertEqual(c.used_units(), 3)

    def test_used_units_after_release(self):
        c = FIFOCache(6)
        c.put("a", 1, cost=2)
        c.put("b", 2, cost=2)
        c.delete("a")
        self.assertEqual(c.used_units(), 2)
        self.assertTrue(c.put("c", 3, cost=4))
        self.assertEqual(c.used_units(), 6)

    def test_growing_entry_can_evict_itself(self):
        c = FIFOCache(4)
        c.put("a", 1, cost=2)
        c.put("b", 2, cost=2)
        self.assertFalse(c.put("a", 9, cost=4))
        self.assertNotIn("a", c)
        self.assertLessEqual(c.used_units(), 4)

    def test_invalid_arguments_raise(self):
        c = LRUCache(2)
        self.assertRaises(ValueError, c.put, "k", 1, 0)
        self.assertRaises(ValueError, c.put, "k", None)
        self.assertRaises(ValueError, LRUCache, -1)

    def test_cost_one_matches_count_capacity(self):
        c = LRUCache(3)
        for key in "abcd":
            c.put(key, key)
        self.assertEqual(c.used_units(), 3)
        self.assertEqual(len(c), 3)


# --------------------------------------------------------------------------- trace
class TestTraces(unittest.TestCase):
    def test_uniform_length_and_bounds(self):
        trace = uniform(random.Random(41), 500, 60)
        self.assertEqual(len(trace), 500)
        self.assertTrue(all(0 <= k < 60 for k in trace))

    def test_seeded_reproducibility(self):
        one = zipf(random.Random(42), 300, 50, 0.99)
        two = zipf(random.Random(42), 300, 50, 0.99)
        self.assertEqual(one, two)
        self.assertNotEqual(one, zipf(random.Random(43), 300, 50, 0.99))

    def test_zipf_skew_increases_with_s(self):
        def top_share(s):
            trace = zipf(random.Random(44), 4000, 200, s)
            counts = {}
            for key in trace:
                counts[key] = counts.get(key, 0) + 1
            hottest = sum(sorted(counts.values(), reverse=True)[:5])
            return hottest / len(trace)

        self.assertGreater(top_share(1.2), top_share(0.7))

    def test_cyclic_scan_structure(self):
        trace = make_cyclic(10, 25, 2)
        self.assertEqual(trace[:25], list(range(10, 35)))
        self.assertEqual(trace[25:35], list(range(10)))

    def test_time_shift_length_and_bounds(self):
        trace = time_shift(random.Random(45), 400, 120, 3, 10)
        self.assertEqual(len(trace), 400)
        self.assertTrue(all(0 <= k < 120 for k in trace))

    def test_time_shift_old_hot_keys_cool_down(self):
        trace = time_shift(random.Random(46), 900, 120, 3, 20)
        first, last = trace[:300], trace[-300:]
        self.assertGreater(first.count(5) + first.count(9), last.count(5) + last.count(9))

    def test_powerlaw_tail_shrinks_as_threshold_grows(self):
        trace = powerlaw_reuse(random.Random(47), 800, 1200, 0.8, 6)
        self.assertGreater(gap_fraction_above(trace, 16), gap_fraction_above(trace, 200))
        self.assertGreater(gap_fraction_above(trace, 200), 0.0)

    def test_reuse_distances_and_percentile(self):
        self.assertEqual(reuse_distances([1, 2, 1]), [2])
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2)
        self.assertEqual(percentile([], 0.9), 0)

    def test_summarize_reports_counts(self):
        info = summarize([1, 2, 1, 3, 2])
        self.assertEqual(info["ops"], 5)
        self.assertEqual(info["unique"], 3)
        self.assertEqual(info["reuses"], 2)


def make_cyclic(hot, scan, reuses):
    return cyclic_scan(random.Random(99), hot * reuses + scan, hot, scan, reuses)


# --------------------------------------------------------------------------- 内存口径
class TestMemory(unittest.TestCase):
    def test_structure_bytes_positive_and_rows(self):
        c = LRUCache(32)
        rows = detail_rows(c)
        self.assertTrue(rows)
        self.assertGreater(structure_bytes(c), 0)
        self.assertTrue(all(label and nbytes > 0 for label, nbytes in rows))

    def test_per_entry_bytes_stable_across_capacity(self):
        small = per_entry_bytes(LRUCache(128))
        large = per_entry_bytes(LRUCache(512))
        self.assertGreater(large, small * 0.5)
        self.assertLess(large, small * 2.0)

    def test_heavier_policies_cost_more(self):
        fifo = structure_bytes(FIFOCache(64))
        two_q = structure_bytes(TwoQCache(64))
        lfu = structure_bytes(LFUCache(64))
        self.assertGreater(two_q, fifo)
        self.assertGreater(lfu, fifo)

    def test_payload_bytes_counts_resident_keys(self):
        c = LRUCache(8)
        c.put(100000, 100000)
        self.assertEqual(payload_bytes(c), int_object_bytes(100000))

    def test_int_object_bytes_plausible(self):
        self.assertGreaterEqual(int_object_bytes(4096), 28)

    def test_owned_containers_names(self):
        names = owned_containers(LFUCache(8))
        self.assertIn("lfu.freq", names)
        self.assertIn("store.map", names)


# --------------------------------------------------------------------------- 一致性
class TestReferenceAgreement(unittest.TestCase):
    def assert_agrees(self, policy, trace, capacity):
        c = build(policy, capacity, trace)
        r = REFS[policy](capacity, trace)
        for key in trace:
            if c.get(key) is None:
                c.put(key, key)
            if r.get(key) is None:
                r.put(key, key)
            self.assertEqual(set(c), r.residents(), "%s @ %r" % (policy, key))
            self.assertEqual(c.hits, r.hits)
            self.assertEqual(c.last_evicted, r.last_evicted)

    def test_fifo_matches_reference(self):
        self.assert_agrees("fifo", sample_traces(600)["zipf"], 16)

    def test_lru_matches_reference(self):
        self.assert_agrees("lru", sample_traces(600)["zipf"], 16)

    def test_two_q_matches_reference(self):
        self.assert_agrees("two_q", sample_traces(600)["scan"], 16)

    def test_lfu_matches_reference(self):
        self.assert_agrees("lfu", sample_traces(600)["shift"], 16)

    def test_belady_matches_reference(self):
        self.assert_agrees("belady", sample_traces(600)["heavy"], 16)

    def test_agrees_on_all_traces(self):
        for name, trace in sample_traces(500).items():
            for policy in POLICIES:
                c = build(policy, 24, trace)
                r = REFS[policy](24, trace)
                feed(c, trace)
                feed(r, trace)
                self.assertEqual(c.hits, r.hits, "%s/%s" % (name, policy))
                self.assertEqual(set(c), r.residents(), "%s/%s" % (name, policy))


class TestInvariants(unittest.TestCase):
    def setUp(self):
        self.traces = sample_traces(700)

    def test_capacity_never_exceeded(self):
        for policy in POLICIES:
            for capacity in (0, 1, 2, 25):
                c = build(policy, capacity, self.traces["zipf"])
                feed(c, self.traces["zipf"])
                self.assertLessEqual(len(c), capacity)
                self.assertLessEqual(c.used_units(), max(capacity, 0))

    def test_never_returns_an_evicted_key(self):
        for policy in POLICIES:
            c = build(policy, 18, self.traces["uniform"])
            gone = set()
            for key in self.traces["uniform"]:
                c.last_evicted = None
                value = c.get(key)
                if c.last_evicted is not None:
                    gone.add(c.last_evicted)
                if value is None:
                    c.put(key, key)
                    if c.last_evicted is not None:
                        gone.add(c.last_evicted)
                    gone.discard(key)
                else:
                    self.assertNotIn(key, gone, policy)

    def test_stats_are_consistent(self):
        for policy in POLICIES:
            c = build(policy, 20, self.traces["shift"])
            hits = feed(c, self.traces["shift"])
            self.assertEqual(hits, c.hits)
            self.assertEqual(c.lookups, len(self.traces["shift"]))
            self.assertAlmostEqual(c.hit_rate, c.hits / float(c.lookups))

    def test_eviction_count_matches_store_release(self):
        for policy in POLICIES:
            c = build(policy, 16, self.traces["heavy"])
            feed(c, self.traces["heavy"])
            self.assertEqual(c.evictions, c.store.releases)

    def test_clear_stats_resets(self):
        c = build("lru", 4, self.traces["zipf"])
        feed(c, self.traces["zipf"])
        c.clear_stats()
        self.assertEqual((c.hits, c.misses, c.evictions), (0, 0, 0))
        self.assertEqual(c.hit_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
