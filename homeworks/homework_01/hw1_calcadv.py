#!/usr/bin/env python
# coding: utf-8


def advanced_calculator(input_string):
    '''
    Калькулятор на основе обратной польской записи.
    Разрешенные операции: открытая скобка, закрытая скобка,
     плюс, минус, умножить, делить
    :param input_string: строка, содержащая выражение
    :return: результат выполнение операции, если строка валидная - иначе None
    '''
    try:
        if '[' or ']' or '{' or '}' or '()' in input_string:
            return None
        else:
            return eval(input_string)
    except:
        return None
