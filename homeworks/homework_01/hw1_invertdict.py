#!/usr/bin/env python
# coding: utf-8

def unpack(x, m):
    if type(x) == list or type(x) == tuple or type(x) == set:
        for i in x:
            if type(i) == list or type(i) == tuple or type(i) == set:
                m = (unpack(i, m))
            else:
                m.append(i)
    else:
        m.append(x)
    return m


def invert_dict(source_dict):
    '''
    Функция которая разворачивает словарь, т.е.
    каждому значению ставит в соответствие ключ.
    :param source_dict: dict
    :return: new_dict: dict
    '''
    new_dict = {}
    for i, j in source_dict.items():
        for k in unpack(j, []):
            if k in new_dict:
                if type(new_dict[k]) == list:
                    new_dict[k].append(i)
                else:
                    new_dict[k] = [new_dict[k]]
                    new_dict[k].append(i)
            else:
                new_dict[k] = i
    return new_dict
