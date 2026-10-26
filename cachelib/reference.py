"""朴素参考实现：用 dict/list 把同样的语义重做一遍，供不变量测试与基准自检对照。

这里刻意不优化——O(n) 的查找、淘汰、排序全都留着。它的价值在于是"另一条独立的
错误路径"：如果槽位存储或链表指针写错了，两边的驻留集合会在某一步分叉；快反而是
缺点，所以每个方法都按最直白的写法走。语义与正式实现严格对齐：每个 get 推进一次
查找计数、写更新算一次访问、同频次先淘汰最久没被碰的。
"""

from collections import deque


class Base:
    def __init__(self, capacity):
        self.capacity = capacity
        self.vals = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected = 0
        self.last_evicted = None

    def residents(self):
        return set(self.vals)

    @property
    def lookups(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return self.hits / self.lookups if self.lookups else 0.0

    def get(self, key, default=None, now=None):
        self._on_lookup()
        if key in self.vals:
            self.hits += 1
            self._read(key)
            return self.vals[key]
        self.misses += 1
        return default

    def put(self, key, value, cost=1, ttl=None, now=None):
        if cost > self.capacity:
            self.rejected += 1
            return False
        if key in self.vals:
            self.vals[key] = value
            self._read(key)
            return True
        if self.capacity <= 0:
            return False
        while len(self.vals) + cost > self.capacity:
            self._evict_one()
        self._admit(key)
        self.vals[key] = value
        return True

    def _on_lookup(self):
        return None

    def _read(self, key):
        return None

    def _admit(self, key):
        return None

    def _evict_one(self):
        raise NotImplementedError


class RefFIFO(Base):
    def __init__(self, capacity):
        Base.__init__(self, capacity)
        self.order = []

    def _admit(self, key):
        self.order.append(key)

    def _evict_one(self):
        victim = self.order.pop(0)
        del self.vals[victim]
        self.last_evicted = victim
        self.evictions += 1


class RefLRU(Base):
    def __init__(self, capacity):
        Base.__init__(self, capacity)
        self.order = []

    def _touch(self, key):
        self.order.remove(key)
        self.order.append(key)

    def _read(self, key):
        self._touch(key)

    def _admit(self, key):
        self.order.append(key)

    def _evict_one(self):
        victim = self.order.pop(0)
        del self.vals[victim]
        self.last_evicted = victim
        self.evictions += 1


class RefTwoQ(Base):
    def __init__(self, capacity, alpha=0.25, gamma=0.5):
        Base.__init__(self, capacity)
        self.prob = []
        self.main = []
        self.prob_budget = min(capacity, max(1, int(alpha * capacity))) if capacity else 0
        self.main_budget = self.capacity - self.prob_budget
        self.ghost_budget = int(gamma * capacity)
        self.ghosts = deque()
        self.ghost_set = set()

    def _note_ghost(self, key):
        if not self.ghost_budget:
            return
        if key in self.ghost_set:
            self.ghosts.remove(key)
        self.ghosts.append(key)
        self.ghost_set.add(key)
        while len(self.ghosts) > self.ghost_budget:
            self.ghost_set.discard(self.ghosts.popleft())

    def _where(self, key):
        if key in self.main:
            return "main"
        if key in self.prob:
            return "prob"
        return None

    def _rebalance(self):
        while len(self.prob) > self.prob_budget:
            self._forget(self.prob.pop(0))
        while len(self.main) > self.main_budget:
            self._forget(self.main.pop(0))

    def _forget(self, victim):
        del self.vals[victim]
        self.last_evicted = victim
        self.evictions += 1
        self._note_ghost(victim)

    def _read(self, key):
        if self._where(key) == "main":
            self.main.remove(key)
            self.main.append(key)
            return
        if self.main_budget:
            self.prob.remove(key)
            self.main.append(key)
        else:
            self.prob.remove(key)
            self.prob.append(key)
        self._rebalance()

    def _admit(self, key):
        if self.main_budget and key in self.ghost_set:
            self.ghost_set.discard(key)
            self.ghosts.remove(key)
            self.main.append(key)
        else:
            self.prob.append(key)
        self._rebalance()

    def _evict_one(self):
        # 正式实现的 probation 用 head=最新、tail=最旧，这里列表尾部是最新，
        # 所以两条都从下标 0 取牺牲品；_rebalance 里同理。
        pool = self.prob if self.prob else self.main
        self._forget(pool.pop(0))


class RefLFU(Base):
    def __init__(self, capacity, decay=0.5, interval=1000):
        Base.__init__(self, capacity)
        self.decay = float(decay)
        self.interval = int(interval)
        self.freq = {}
        self.last = {}
        self.ticks = 0
        self.stamp = 0

    def _on_lookup(self):
        self.ticks += 1
        if self.decay < 1.0 and self.ticks >= self.interval:
            self._sweep()

    def _sweep(self):
        self.ticks = 0
        for key in list(self.freq):
            self.freq[key] = int(self.freq[key] * self.decay)

    def _read(self, key):
        self.freq[key] = self.freq.get(key, 0) + 1
        self.stamp += 1
        self.last[key] = self.stamp

    def _admit(self, key):
        self.freq[key] = 1
        self.stamp += 1
        self.last[key] = self.stamp

    def _evict_one(self):
        best = None
        for key in self.vals:
            if best is None:
                best = key
            elif self.freq[key] < self.freq[best]:
                best = key
            elif self.freq[key] == self.freq[best] and self.last[key] < self.last[best]:
                best = key
        del self.vals[best]
        del self.freq[best]
        del self.last[best]
        self.last_evicted = best
        self.evictions += 1


class RefOptimal(Base):
    """MIN 的另一条独立实现：从当前位置向后线性扫描找下一次使用。"""

    def __init__(self, capacity, trace):
        Base.__init__(self, capacity)
        self.trace = trace
        self.pos = -1
        self.due = {}
        self.stamp = {}
        self.counter = 0

    def _next_use(self, key):
        for p in range(self.pos + 1, len(self.trace)):
            if self.trace[p] == key:
                return p
        return None

    def _on_lookup(self):
        self.pos += 1

    def _read(self, key):
        self.counter += 1
        self.due[key] = self._next_use(key)
        self.stamp[key] = self.counter

    def _admit(self, key):
        self.counter += 1
        self.due[key] = self._next_use(key)
        self.stamp[key] = self.counter

    def _evict_one(self):
        # "以后再也不会用到"并列时按最久没被碰的先淘汰，和正式实现的堆序一致；
        # 并列怎么破不影响命中数（都是最优），只影响驻留集合，所以要显式对齐才能逐键比对。
        best = None
        best_due = -1
        best_stamp = None
        for key in self.vals:
            due = self.due[key]
            value = len(self.trace) + 1 if due is None else due
            if value > best_due or (value == best_due and self.stamp[key] < best_stamp):
                best_due = value
                best_stamp = self.stamp[key]
                best = key
        del self.vals[best]
        del self.due[best]
        del self.stamp[best]
        self.last_evicted = best
        self.evictions += 1
