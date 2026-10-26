"""LFU + 老化：淘汰频次最低者，每 interval 次查找把所有频次乘一次 decay。

老化是这个策略真正的旋钮，不是装饰。关掉老化（decay=1.0）后，早期刷进去的高频
键会永久占位，后来的热键进不来；开得太狠（decay→0）频次被抹平，退化成本地 LRU。
两个方向都要实测，才知道中间哪里最划算。
"""

import heapq

from .base import Cache


class LFUCache(Cache):
    def __init__(self, capacity, decay=0.5, interval=1000, time_fn=None):
        super().__init__(capacity, time_fn)
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay 必须落在 [0, 1] 内")
        if interval <= 0:
            raise ValueError("interval 必须是正整数")
        self.decay = float(decay)
        self.interval = int(interval)
        self.freq = self.store.make_array(0)
        self.last = self.store.make_array(0)
        self.buckets = {}
        self.heap = []
        self.ticks = 0
        self.stamp = 0
        self.sweeps = 0

    # ------------------------------------------------------------------ 老化
    def _on_lookup(self, key):
        # 老化以 lookup 为单位推进，因此同一条 trace 上 decay/interval 的组合是确定的。
        self.ticks += 1
        if self.decay < 1.0 and self.ticks >= self.interval:
            self._sweep()

    def _sweep(self):
        """把所有频次乘 decay 并按新频次重建桶。

        分两步走：先只读地按新频次分组，再统一重挂链表。一边遍历旧桶一边改指针
        会把还没遍历到的旧桶链接踩掉，因为所有桶共用同一对 prev/nxt 数组。
        桶内按 last 降序入列，保证 _victim 取 tail 时拿到的仍是"最久没被碰"的那条。
        """
        self.ticks = 0
        self.sweeps += 1
        groups = {}
        for i in list(self.store.map.values()):
            nf = int(self.freq[i] * self.decay)
            self.freq[i] = nf
            groups.setdefault(nf, []).append(i)
        fresh = {}
        for nf, idxs in groups.items():
            idxs.sort(key=self.last.__getitem__, reverse=True)
            lst = self.store.make_list()
            for i in idxs:
                lst.push_back(i)
            fresh[nf] = lst
        self.buckets = fresh
        self.heap = list(fresh)
        heapq.heapify(self.heap)

    def _bucket(self, f):
        lst = self.buckets.get(f)
        if lst is None:
            lst = self.store.make_list()
            self.buckets[f] = lst
            # 空桶在 _remove/_victim 时会被整个删掉，所以走到这里的一定是新频次，
            # 必须登记进堆；否则堆里的频次一旦被弹掉就永久消失，_victim 会返回 -1。
            heapq.heappush(self.heap, f)
        return lst

    def _leave(self, f, i):
        lst = self.buckets.get(f)
        if lst is None:
            return
        lst.unlink(i)
        if lst.length == 0:
            del self.buckets[f]

    # ------------------------------------------------------------------ 钩子
    def _install(self, i):
        self.freq[i] = 1
        self.stamp += 1
        self.last[i] = self.stamp
        self._bucket(1).push_front(i)

    def _touch(self, i):
        f = self.freq[i]
        self._leave(f, i)
        nf = f + 1
        self.freq[i] = nf
        # last 用严格递增的 stamp 而不是 lookup 计数：写更新不会推进 lookup 计数，
        # 两者混用会让同频次桶里出现并列，桶内顺序就变成实现细节而不是语义。
        self.stamp += 1
        self.last[i] = self.stamp
        self._bucket(nf).push_front(i)

    def _remove(self, i):
        self._leave(self.freq[i], i)

    def _victim(self):
        while self.heap:
            f = self.heap[0]
            lst = self.buckets.get(f)
            if lst is not None and lst.length:
                # 只看不弹：真正的释放由基类 _drop -> _remove 完成，
                # 桶空了之后这条记录会在下一次 _victim 时被弹掉。
                return lst.tail
            heapq.heappop(self.heap)
        return -1

    def frequencies(self):
        out = {}
        for key, i in self.store.map.items():
            out[key] = self.freq[i]
        return out

    def bytes_detail(self):
        rows = Cache.bytes_detail(self)
        rows.append(("lfu.freq", self.freq))
        rows.append(("lfu.last", self.last))
        rows.append(("lfu.buckets", self.buckets))
        rows.append(("lfu.heap", self.heap))
        for f, lst in self.buckets.items():
            rows.append(("lfu.bucket#%d" % f, lst))
        return rows
