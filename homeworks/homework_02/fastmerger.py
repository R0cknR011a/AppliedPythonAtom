#!/usr/bin/env python
# coding: utf-8

from .heap import MaxHeap


class FastSortedListMerger:

    @staticmethod
    def merge_first_k(list_of_lists, k):
        '''
        принимает на вход список отсортированных непоубыванию списков и число
        на выходе выдает один список длинной k, отсортированных по убыванию
        '''
        m = MaxHeap([])
        result = []
        print(list_of_lists, k)
        for i in list_of_lists:
            for j in i:
                m.add((j, list_of_lists.index(i)))
        print(m.heap)
        for i in range(k):
            result.append(m.extract_maximum()[0])
        return result
