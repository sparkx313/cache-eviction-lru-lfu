"""LRU：最近被读写过的留在 head，淘汰 tail。

对局部性好的负载几乎总是默认答案，代价是每个命中都要做一次 unlink+push_front；
对整段顺序扫描，它会把刚建立的热点全部让位给只见过一次的键。
"""

from .base import Cache


class LRUCache(Cache):
    def __init__(self, capacity, time_fn=None):
        super().__init__(capacity, time_fn)
        self.order = self.store.make_list()

    def _install(self, i):
        self.order.push_front(i)

    def _touch(self, i):
        self.order.move_to_front(i)

    def _remove(self, i):
        self.order.unlink(i)

    def _victim(self):
        return self.order.tail

    def resident_order(self):
        return [self.store.keys[i] for i in self.order.indexes()]
