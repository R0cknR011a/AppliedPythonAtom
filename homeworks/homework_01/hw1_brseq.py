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
    if input_string == '':
        return True
    if input_string[0] in ')]}':
        return False
    if input_string[-1] in '([{':
        return False
    for i in range(len(input_string)-1):
        if input_string[i] == '(':
            if input_string[i+1] == ']' or input_string[i+1] == '}':
                return False
        if input_string[i] == '[':
            if input_string[i+1] == ')' or input_string[i+1] == '}':
                return False
        if input_string[i] == '{':
            if input_string[i+1] == ')' or input_string[i+1] == ']':
                return False
    if input_string.count('(') == input_string.count(')'):
        if input_string.count('[') == input_string.count(']'):
            if input_string.count('{') == input_string.count('}'):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
