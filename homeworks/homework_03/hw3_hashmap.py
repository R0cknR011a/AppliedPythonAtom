#!/usr/bin/env python
# coding: utf-8


class HashMap:
    '''
    Давайте сделаем все объектненько,
     поэтому внутри хешмапы у нас будет Entry
    '''
    class Entry:
        def __init__(self, key, value):
            '''
            Сущность, которая хранит пары ключ-значение
            :param key: ключ
            :param value: значение
            '''
            self.key = key
            self.value = value

        def get_key(self):
            # TODO возвращаем ключ
            return self.key

        def get_value(self):
            # TODO возвращаем значение
            return self.value

        def __eq__(self, other):
            # TODO реализовать функцию сравнения
            return self.key == other.key

    def __init__(self, bucket_num=64):
        '''
        Реализуем метод цепочек
        :param bucket_num: число бакетов при инициализации
        '''
        self.buckets = [None] * bucket_num

    def get(self, key, default_value=None):
        # TODO метод get, возвращающий значение,
        #  если оно присутствует, иначе default_value
        if self.buckets[self._get_index(self._get_hash(key))] is not None:
            for i in self.buckets[self._get_index(self._get_hash(key))]:
                if i.get_key() == key:
                    return i.get_value()
        return default_value

    def put(self, key, value):
        # TODO метод put, кладет значение по ключу,
        #  в случае, если ключ уже присутствует он его заменяет
        if self.buckets[self._get_index(self._get_hash(key))] is None:
            self.buckets[self._get_index(self._get_hash(key))] = [self.Entry(key, value)]
        elif self.Entry(key, value) in self.buckets[self._get_index(self._get_hash(key))]:
            for i in self.buckets[self._get_index(self._get_hash(key))]:
                if i.key == key:
                    i.value = value
        else:
            self.buckets[self._get_index(self._get_hash(key))].append(self.Entry(key, value))

    def __len__(self):
        # TODO Возвращает количество Entry в массиве
        counter = 0
        for i in self.buckets:
            if i is not None:
                counter += len(i)
        return counter

    def _get_hash(self, key):
        # TODO Вернуть хеш от ключа,
        #  по которому он кладется в бакет
        return hash(key)

    def _get_index(self, hash_value):
        # TODO По значению хеша вернуть индекс элемента в массиве
        return hash_value % len(self.buckets)

    def values(self):
        # TODO Должен возвращать итератор значений
        return [j.value for i in self.buckets if i is not None
                for j in i]

    def keys(self):
        # TODO Должен возвращать итератор ключей
        return [j.key for i in self.buckets if i is not None
                for j in i]

    def items(self):
        # TODO Должен возвращать итератор пар ключ и значение (tuples)
        return [(j.key, j.value) for i in self.buckets if i is not None
                for j in i]

    def _resize(self):
        # TODO Время от времени нужно ресайзить нашу хешмапу
        self.buckets += [None] * (len(self.buckets) // 2)

    def __str__(self):
        # TODO Метод выводит "buckets: {}, items: {}"
        return "buckets: {}, items: {}".format(self.buckets, self.items())

    def __contains__(self, item):
        # TODO Метод проверяющий есть ли объект (через in)
        return item in self.keys()
