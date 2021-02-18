#!/usr/bin/env python
# coding: utf-8

from homeworks.homework_03.hw3_hashmap import HashMap


class HashSet(HashMap):

    def __init__(self):
        # TODO Сделать правильно =)
        super().__init__()

    def get(self, key, default_value=None):
        # TODO достаточно переопределить данный метод
        return super().__contains__(key)

    def put(self, key, value=None):
        # TODO метод put, нужно переопределить данный метод
        return super().put(key, value)

    def __len__(self):
        # TODO Возвращает количество Entry в массиве
        return super().__len__()

    def values(self):
        # TODO возвращать итератор значений
        return super().keys()

    def intersect(self, another_hashset):
        # TODO метод, возвращающий новый HashSet
        #  элементы - пересечение текущего и другого
        new_hashset = HashSet()
        values = self.values() + another_hashset.values()
        for i in values:
            if i in self.values() and i in another_hashset.values():
                new_hashset.put(i)
        return new_hashset
