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
        if input_string in ('', '*', '/', '()', ',', '[', ']', '**'):
            return None
        for i in input_string:
            if i.isalpha() or i in '[' or i in ']' or i in ',':
                return None
            if '**' in input_string:
                return None
        return eval(input_string)
    except SyntaxError:
        return None
