#!/usr/bin/env python
# coding: utf-8


def reverse(number):
    '''
    Метод, принимающий на вход int и
    возвращающий инвертированный int
    :param number: исходное число
    :return: инвертированное число
    '''
    positive = False
    string_number = ''
    if number != 0:
        if number > 0:
            positive = True
        number = abs(number)
        while number % 10 == 0:
            number = number / 10
        string_number = (str(int(number)))[::-1]
        return string_number if positive else '-' + string_number
    else:
        return 0
    
