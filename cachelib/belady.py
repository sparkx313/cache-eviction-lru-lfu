"""MIN/Belady：用整条 trace 的未来信息，淘汰"最久以后才要用"的条目。

它不能上线——线上没有未来——它的唯一用处是给出命中率的理论上界，好把每个启发式
的差距换算成"离可达上限还差多少"。堆里存 `-next_use`，因此"以后再也不会用到"
（INF）的条目最先被淘汰，这正是 MIN 的定义。
"""

import bisect
import heapq

from .base import Cache

INF = float("inf")


class BeladyCache(Cache):
    def __init__(self, capacity, trace, time_fn=None):
        super().__init__(capacity, time_fn)
        self.trace = trace
        self._positions = {}
        for pos, key in enumerate(trace):
            bucket = self._positions.get(key)
            if bucket is None:
                self._positions[key] = [pos]
            else:
                bucket.append(pos)
        self.cursor = -1
        self._due = self.store.make_array(INF)
        self._stamp = self.store.make_array(0)
        self._heap = []
        self._seq = 0

    def _on_lookup(self, key):
        # 依赖 feed() 的形状：一次 lookup 恰好推进一个 trace 位置。
        self.cursor += 1

    def _next_use(self, key):
        positions = self._positions.get(key)
        if positions is None:
            return INF
        p = bisect.bisect_right(positions, self.cursor)
        return positions[p] if p < len(positions) else INF

    def _push(self, i):
        self._seq += 1
        self._due[i] = self._next_use(self.store.keys[i])
        self._stamp[i] = self._seq
        heapq.heappush(self._heap, (-self._due[i], self._seq, i))

    def _install(self, i):
        self._push(i)

    def _touch(self, i):
        self._push(i)

    def _remove(self, i):
        self._stamp[i] = -1

    def _victim(self):
        while self._heap:
            neg, seq, i = self._heap[0]
            if seq == self._stamp[i] and self.store.keys[i] is not None:
                return i
            heapq.heappop(self._heap)
        return -1

    def resident_order(self):
        rows = [(self._due[i], key) for key, i in self.store.map.items()]
        rows.sort()
        return [key for _, key in rows]

    def bytes_detail(self):
        rows = Cache.bytes_detail(self)
        rows.append(("belady.due", self._due))
        rows.append(("belady.stamp", self._stamp))
        rows.append(("belady.heap", self._heap))
        rows.append(("belady.positions", self._positions))
        return rows
