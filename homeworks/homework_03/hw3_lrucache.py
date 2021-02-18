#!/usr/bin/env python
# coding: utf-8
import time
from collections import OrderedDict


class LRUCacheDecorator:

    def __init__(self, maxsize, ttl):
        '''
        :param maxsize: максимальный размер кеша
        :param ttl: время в млсек, через которое кеш
                    должен исчезнуть
        '''
        # TODO инициализация декоратора
        #  https://www.geeksforgeeks.org/class-as-decorator-in-python/
        self.maxsize = maxsize
        self.dict = OrderedDict()
        self.start = time.time()
        if ttl is None:
            self.ttl = 9999999
        else:
            self.ttl = ttl

    def __call__(self, f):
        # TODO вызов функции
        def func(*args, **kwargs):
            m = []
            for i in args:
                m.append(i)
            for i, j in kwargs.items():
                m.append(i, j)
            input = ''.join(str(x) for x in m)
            if time.time() - self.start < self.ttl:
                if input in self.dict:
                        if len(self.dict) == self.maxsize:
                            x = self.dict[input]
                            del self.dict[input]
                            self.dict[input] = x
                        return self.dict[input]
                else:
                    if len(self.dict) == self.maxsize:
                        delete = list(self.dict.keys())[0]
                        del self.dict[delete]
                    self.dict[input] = f(*args, **kwargs)
                    return f(*args, **kwargs)
            else:
                self.dict.clear()
                self.start = time.time()
                self.dict[input] = f(*args, **kwargs)
                return f(*args, **kwargs)
        return func
