"""FIFO：只按插入顺序淘汰，读命中不改变任何顺序。

它是所有比较的下界参照：实现最便宜，但对"热键会被反复读"这件事完全无感。
"""

from .base import Cache


class FIFOCache(Cache):
    def __init__(self, capacity, time_fn=None):
        super().__init__(capacity, time_fn)
        self.queue = self.store.make_list()

    def _install(self, i):
        self.queue.push_back(i)

    def _touch(self, i):
        # 不重排是特性不是缺陷：这条决定让 FIFO 和 LRU 的全部差别只剩一次 unlink。
        return None

    def _remove(self, i):
        self.queue.unlink(i)

    def _victim(self):
        return self.queue.head

    def resident_order(self):
        return [self.store.keys[i] for i in self.queue.indexes()]
