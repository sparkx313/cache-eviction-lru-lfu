"""字节口径：只统计容器自身的 sys.getsizeof，不递归进被引用的对象。

list 的 getsizeof 只含指针数组，dict 只含哈希表骨架，所以这张表回答的是
"这条数据结构自己占了多少"。键与值对象本身由 trace 持有、不计入结构列，
要连负载一起看就单独调 payload_bytes()。
"""

import sys


def detail_rows(cache):
    """返回 [(容器名, 字节)]，逐个走一遍 cache.bytes_detail()。"""
    return [(label, sys.getsizeof(obj)) for label, obj in cache.bytes_detail()]


def structure_bytes(cache):
    return sum(nbytes for _, nbytes in detail_rows(cache))


def payload_bytes(cache):
    """驻留键对象自身的字节之和；本库里 value 恒等于 key，所以只计一份。"""
    total = 0
    for key in cache:
        total += sys.getsizeof(key)
    return total


def int_object_bytes(sample=4096):
    return sys.getsizeof(sample)


def per_entry_bytes(cache):
    """结构字节 / 容量：按"装满"折算每条目的固定开销，与当前驻留数无关。"""
    if cache.capacity <= 0:
        return 0.0
    return structure_bytes(cache) / float(cache.capacity)


def owned_containers(cache):
    return [label for label, _ in cache.bytes_detail()]
