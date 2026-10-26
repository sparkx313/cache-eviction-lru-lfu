"""紧凑槽位存储：所有策略共用的一套自研结构，不依赖 OrderedDict。

槽位就是 0..capacity-1 的下标。四个平行数组存 key/value/cost/exps，一个 dict
做 key -> slot 的反查；淘汰顺序由策略自己的链表或堆决定，存储层不管策略。

空闲槽位借用 exps 数组串成一条单向链，因此分配与释放都是 O(1)，也省掉第五个
数组：代价是任何地方都不能去读一个空闲槽的 exps。
"""


class SlotList:
    """在宿主 prev/nxt 平行数组上操作的入侵式双向链表，-1 表示空指针。

    head 端与 tail 端谁是"最新"由宿主决定：LRU 用 push_front + victim=tail，
    FIFO 用 push_back + victim=head。同一个数组能同时服务多条链表，因为任一
    时刻一个槽位只属于一条链表。
    """

    __slots__ = ("prev", "nxt", "head", "tail", "length")

    def __init__(self, prev, nxt):
        self.prev = prev
        self.nxt = nxt
        self.head = -1
        self.tail = -1
        self.length = 0

    def __len__(self):
        return self.length

    def push_front(self, i):
        prev, nxt = self.prev, self.nxt
        prev[i] = -1
        nxt[i] = self.head
        if self.head >= 0:
            prev[self.head] = i
        else:
            self.tail = i
        self.head = i
        self.length += 1

    def push_back(self, i):
        prev, nxt = self.prev, self.nxt
        nxt[i] = -1
        prev[i] = self.tail
        if self.tail >= 0:
            nxt[self.tail] = i
        else:
            self.head = i
        self.tail = i
        self.length += 1

    def unlink(self, i):
        prev, nxt = self.prev, self.nxt
        p = prev[i]
        n = nxt[i]
        if p >= 0:
            nxt[p] = n
        else:
            self.head = n
        if n >= 0:
            prev[n] = p
        else:
            self.tail = p
        prev[i] = -1
        nxt[i] = -1
        self.length -= 1

    def move_to_front(self, i):
        self.unlink(i)
        self.push_front(i)

    def move_to_back(self, i):
        self.unlink(i)
        self.push_back(i)

    def pop_front(self):
        i = self.head
        if i >= 0:
            self.unlink(i)
        return i

    def pop_back(self):
        i = self.tail
        if i >= 0:
            self.unlink(i)
        return i

    def indexes(self):
        """从 head 到 tail 走一遍，只用于测试与调试。"""
        out = []
        i = self.head
        while i >= 0:
            out.append(i)
            i = self.nxt[i]
        return out


class SlotStore:
    """定长槽位池：满了必须由调用方先淘汰，alloc 从不扩容。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.keys = [None] * capacity
        self.vals = [None] * capacity
        self.costs = [1] * capacity
        self.exps = [None] * capacity
        self.free_head = 0 if capacity else -1
        self.prev = None
        self.nxt = None
        self.map = {}
        self.used = 0
        self.allocs = 0
        self.releases = 0
        if capacity:
            # 初始化空闲链：exps[i] = i+1，末尾 -1 表示链尾。
            for i in range(capacity):
                self.exps[i] = i + 1
            self.exps[capacity - 1] = -1

    def __len__(self):
        return len(self.map)

    @property
    def free_count(self):
        return self.capacity - len(self.map)

    def is_live(self, i):
        return 0 <= i < self.capacity and self.keys[i] is not None

    def alloc(self, key, value, cost=1):
        i = self.free_head
        if i < 0:
            raise RuntimeError("槽位池已满，调用方必须先淘汰一条")
        # 先取走链里的下一个空闲下标，再覆盖 exps 的含义（活槽的 exps 是过期时间）。
        self.free_head = self.exps[i]
        self.keys[i] = key
        self.vals[i] = value
        self.costs[i] = cost
        self.exps[i] = None
        self.map[key] = i
        self.used += cost
        self.allocs += 1
        return i

    def release(self, i):
        key = self.keys[i]
        del self.map[key]
        self.used -= self.costs[i]
        self.keys[i] = None
        self.vals[i] = None
        self.costs[i] = 1
        self.exps[i] = self.free_head
        self.free_head = i
        self.releases += 1
        return key

    def set_cost(self, i, cost):
        self.used += cost - self.costs[i]
        self.costs[i] = cost

    def make_list(self):
        """策略按需申请链表；所有链表共用同一对 prev/nxt 数组。"""
        if self.prev is None:
            self.prev = [-1] * self.capacity
            self.nxt = [-1] * self.capacity
        return SlotList(self.prev, self.nxt)

    def make_array(self, fill=0):
        return [fill] * self.capacity

    def detail(self):
        """返回 (名字, 对象) 列表，供内存口径逐容器统计。"""
        rows = [("store.keys", self.keys), ("store.vals", self.vals),
                ("store.costs", self.costs), ("store.exps", self.exps),
                ("store.map", self.map)]
        if self.prev is not None:
            rows.append(("store.prev", self.prev))
            rows.append(("store.nxt", self.nxt))
        return rows
