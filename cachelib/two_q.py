"""2Q-lite（带年龄分区的 LRU）：新键先进试用期队列，只有被复用过的键才进主队列。

三个部分缺一不可：
  probation  只见过一次的键，容量的一小份，超预算就从最旧端淘汰
  main       被复用过的键，或"曾经被淘汰过、这次又出现"的键，剩下的容量都归它
  ghosts     只存键名、不存值的淘汰记录；命中 ghost 意味着这个键值得直接进 main

ghost 表不是可有可无的：先前的实测里去掉它以后，probation 预算一小就再没有键能晋升，
整个缓存退化成"只装最后一次未命中"，命中率归零。留着它才是 2Q 的本意，代价是要为
键名本身维护一个集合。
"""

from collections import deque

from .base import Cache

PROB = 1
MAIN = 2


class TwoQCache(Cache):
    def __init__(self, capacity, alpha=0.25, gamma=0.5, time_fn=None):
        super().__init__(capacity, time_fn)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha 必须落在 (0, 1) 内")
        if not 0.0 < gamma < 2.0:
            raise ValueError("gamma 必须落在 (0, 2) 内")
        self.probation = self.store.make_list()
        self.main = self.store.make_list()
        # where 记录槽位当前挂在哪条队列上，_touch 与 _remove 都靠它判断；0 表示不在缓存。
        self.where = self.store.make_array(0)
        self.alpha = alpha
        self.gamma = gamma
        self.prob_budget = min(capacity, max(1, int(alpha * capacity))) if capacity else 0
        self.main_budget = self.capacity - self.prob_budget
        self.ghost_budget = int(gamma * capacity)
        self.ghosts = deque()
        self.ghost_set = set()
        self.promotions = 0
        self.ghost_hits = 0

    @property
    def probation_length(self):
        return len(self.probation)

    @property
    def main_length(self):
        return len(self.main)

    @property
    def ghost_length(self):
        return len(self.ghosts)

    # ------------------------------------------------------------------ ghost 表
    def _note_ghost(self, key):
        if not self.ghost_budget:
            return
        if key in self.ghost_set:
            # 同一个键可能有两条 ghost 记录，先删旧的那条，集合与队列才不会错位。
            self.ghosts.remove(key)
        self.ghosts.append(key)
        self.ghost_set.add(key)
        while len(self.ghosts) > self.ghost_budget:
            self.ghost_set.discard(self.ghosts.popleft())

    def _drop_ghost(self, key):
        if key in self.ghost_set:
            self.ghost_set.discard(key)
            self.ghosts.remove(key)

    # ------------------------------------------------------------------ 钩子
    def _install(self, i):
        key = self.store.keys[i]
        if self.main_budget and key in self.ghost_set:
            # ghost 命中说明这个键以前就在缓存里待过，第二次见面直接进主队列，
            # 不必再走一遍 probation——扫描键之所以污染 LRU 就是被"从头排起"害的。
            self._drop_ghost(key)
            self.where[i] = MAIN
            self.main.push_front(i)
            self.ghost_hits += 1
        else:
            self.where[i] = PROB
            self.probation.push_front(i)
        self._rebalance()

    def _touch(self, i):
        if self.where[i] == MAIN:
            self.main.move_to_front(i)
            return
        if self.main_budget:
            # 第二次被读中就晋升。main 预算为 0 时留在 probation 刷新位置，
            # 否则晋升完立刻被 _rebalance 踢掉，等于把刚读到的键扔掉。
            self.probation.unlink(i)
            self.where[i] = MAIN
            self.main.push_front(i)
            self.promotions += 1
        else:
            self.probation.move_to_front(i)
        self._rebalance()

    def _rebalance(self):
        while len(self.probation) > self.prob_budget:
            self._drop(self.probation.tail)
            self.evictions += 1
        while len(self.main) > self.main_budget:
            self._drop(self.main.tail)
            self.evictions += 1

    def _remove(self, i):
        w = self.where[i]
        if w == PROB:
            self.probation.unlink(i)
        elif w == MAIN:
            self.main.unlink(i)
        self.where[i] = 0
        self._note_ghost(self.store.keys[i])

    def _victim(self):
        # 容量不够时先牺牲 probation 最旧端：一次性扫描键的代价不该由 main 的工作集付。
        if len(self.probation):
            return self.probation.tail
        return self.main.tail

    def resident_order(self):
        keys = [self.store.keys[i] for i in self.main.indexes()]
        keys += [self.store.keys[i] for i in self.probation.indexes()]
        return keys

    def bytes_detail(self):
        rows = Cache.bytes_detail(self)
        rows.append(("two_q.where", self.where))
        rows.append(("two_q.ghosts", self.ghosts))
        rows.append(("two_q.ghost_set", self.ghost_set))
        return rows
