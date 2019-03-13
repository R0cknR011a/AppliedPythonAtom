#!/usr/bin/env python
# coding: utf-8


class Heap:

    def __init__(self, array):
        self.heap = array[:]
        self.build_heap()

    def add(self, elem_with_priority):
        self.heap.append(elem_with_priority)
        self.sift_up(len(self.heap) - 1)

    def build_heap(self):
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self.sift_down(i)

    def sift_down(self, i):
        max_index = i
        if 2 * i + 1 < len(self.heap) \
                and comparator_d(self.heap[2 * i + 1], self.heap[max_index]):
            max_index = 2 * i + 1
        if 2 * i + 2 < len(self.heap) \
                and comparator_d(self.heap[2 * i + 2], self.heap[max_index]):
            max_index = 2 * i + 2
        if i != max_index:
            self.heap[i], self.heap[max_index] = \
                self.heap[max_index], self.heap[i]
            self.sift_down(max_index)

    def sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if comparator_d(self.heap[parent], self.heap[i]) is False:
                self.heap[parent], self.heap[i] = \
                    self.heap[i], self.heap[parent]
            i = parent


class MaxHeap(Heap):

    def __init__(self, array):
        super().__init__(array)

    def extract_maximum(self):
        result = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.sift_down(0)
        return result


def comparator_d(x, y):
    if x[0] == y[0]:
        return x[1] >= y[1]
    elif x[0] > y[0]:
        return True
    else:
        return False
