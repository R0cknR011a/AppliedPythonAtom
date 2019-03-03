#!/usr/bin/env python
# coding: utf-8


def is_bracket_correct(input_string):
    '''
    Метод проверяющий является ли поданная скобочная
     последовательность правильной (скобки открываются и закрываются)
     не пересекаются
    :param input_string: строка, содержащая 6 типов скобок (,),[,],{,}
    :return: True or False
    '''
    m = []
    dict = {')': '(', ']': '[', '}': '{'}
    for i in input_string:
        if i in '([{':
            m.append(i)
        if i in ')]}':
            if len(m) == 0:
                return False
            elif m[-1] == dict[i]:
                m.pop()
            else:
                return False
    return len(m) == 0

