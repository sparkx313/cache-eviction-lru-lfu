"""所有策略共享的外壳：容量约束、TTL、加权占用、命中统计与钩子协议。

子类只需实现四个钩子：
  _install(slot) 新条目进入缓存（槽位数据已写好）
  _touch(slot)   已在缓存中的条目被读中或被写更新
  _remove(slot)  条目即将离开缓存（此刻槽位里的数据还能读）
  _victim()      返回应当被淘汰的槽位下标，-1 表示无可淘汰

约定：缓存里的值永不为 None，因此 `get() is None` 就是未命中；trace 生成器
生成的 value 就等于 key 本身，天然满足这条约定。
"""

import time

from .store import SlotStore


class Cache:
    def __init__(self, capacity, time_fn=None):
        if capacity < 0:
            raise ValueError("容量不能为负")
        self.capacity = capacity
        self.store = SlotStore(capacity)
        self.time_fn = time.monotonic if time_fn is None else time_fn
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected = 0
        self.expirations = 0
        self.deletes = 0
        self.last_evicted = None

    # ------------------------------------------------------------------ 基本协议
    def __len__(self):
        return len(self.store.map)

    def __iter__(self):
        return iter(list(self.store.map))

    def __contains__(self, key):
        i = self.store.map.get(key)
        return i is not None and not self._expired(i, self.time_fn())

    def __repr__(self):
        return "<%s capacity=%d resident=%d hits=%d misses=%d>" % (
            type(self).__name__, self.capacity, len(self.store.map),
            self.hits, self.misses)

    @property
    def name(self):
        return type(self).__name__.replace("Cache", "").lower()

    @property
    def lookups(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return self.hits / self.lookups if self.lookups else 0.0

    def clear_stats(self):
        self.hits = self.misses = self.evictions = 0
        self.rejected = self.expirations = self.deletes = 0
        self.last_evicted = None

    def used_units(self):
        """当前占用的容量单位（cost>1 的条目会占多份）。"""
        return self.store.used

    # ------------------------------------------------------------------ 时钟与 TTL
    def _clock(self, now):
        return self.time_fn() if now is None else float(now)

    def _set_exp(self, i, ttl, tick):
        self.store.exps[i] = None if ttl is None else tick + float(ttl)

    def _expired(self, i, tick):
        # 只对活槽调用：空闲槽的 exps 存的是空闲链下标，读它会得到荒谬结论。
        e = self.store.exps[i]
        return e is not None and e <= tick

    def purge_expired(self, now=None):
        tick = self._clock(now)
        doomed = [i for key, i in list(self.store.map.items()) if self._expired(i, tick)]
        for i in doomed:
            self._drop(i)
            self.expirations += 1
        return len(doomed)

    # ------------------------------------------------------------------ get / put
    def get(self, key, default=None, now=None):
        tick = self._clock(now)
        self._on_lookup(key)
        i = self.store.map.get(key)
        if i is not None and self._expired(i, tick):
            self._drop(i)
            self.expirations += 1
            i = None
        if i is None:
            self.misses += 1
            return default
        self.hits += 1
        self._touch(i)
        return self.store.vals[i]

    def put(self, key, value, cost=1, ttl=None, now=None):
        if cost < 1:
            raise ValueError("cost 必须是正整数")
        if value is None:
            raise ValueError("缓存不接受 None 值")
        tick = self._clock(now)
        if cost > self.capacity:
            # 单条目就超过整个容量：永远装不下，直接拒收而不是把缓存清空。
            self.rejected += 1
            return False
        i = self.store.map.get(key)
        if i is not None and self._expired(i, tick):
            self._drop(i)
            self.expirations += 1
            i = None
        if i is not None:
            self.store.vals[i] = value
            self._set_exp(i, ttl, tick)
            self.store.set_cost(i, cost)
            while self.store.used > self.capacity:
                v = self._victim()
                if v < 0:
                    break
                self._drop(v)
                self.evictions += 1
                if v == i:
                    # 把已有条目放大时，它自己可能就是牺牲品；此时这条更新等于淘汰，
                    # 不能再 touch 一个已经进空闲链的槽。
                    return False
            self._touch(i)
            return True
        while self.store.used + cost > self.capacity:
            v = self._victim()
            if v < 0:
                break
            self._drop(v)
            self.evictions += 1
        j = self.store.alloc(key, value, cost)
        self._set_exp(j, ttl, tick)
        self._install(j)
        return True

    def delete(self, key):
        i = self.store.map.get(key)
        if i is None:
            return False
        self._drop(i)
        self.deletes += 1
        return True

    def _drop(self, i):
        key = self.store.keys[i]
        self._remove(i)
        self.store.release(i)
        self.last_evicted = key
        return key

    # ------------------------------------------------------------------ 钩子默认值
    def _on_lookup(self, key):
        return None

    def _install(self, i):
        raise NotImplementedError

    def _touch(self, i):
        raise NotImplementedError

    def _remove(self, i):
        raise NotImplementedError

    def _victim(self):
        raise NotImplementedError

    def bytes_detail(self):
        return self.store.detail()


def feed(cache, trace):
    """读穿透驱动：每个 trace 条目恰好一次 get，未命中就立刻 put。

    BeladyCache 的游标推进依赖这个形状（一次 lookup 对应 trace 的一个位置），
    所以所有命中率比较都必须走这里，不能自己手写循环。
    """
    for key in trace:
        if cache.get(key) is None:
            cache.put(key, key)
    return cache.hits


def hit_rate(cache, trace):
    c = cache() if callable(cache) else cache
    feed(c, trace)
    return c.hit_rate
