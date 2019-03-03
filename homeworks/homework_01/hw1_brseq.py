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
    for i in range(len(input_string)):
        if input_string[i] == '(':
            if input_string[i+1] == ']' or '}':
                return False
        if input_string[i] == '[':
            if input_string[i+1] == ')' or '}':
                return False
        if input_string[i] == '{':
            if input_string[i+1] == ')' or ']':
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
