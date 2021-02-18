#!/usr/bin/env python
# coding: utf-8


def reverse(number):
    """
    Метод, принимающий на вход int и
    возвращающий инвертированный int
    :param number: исходное число
    :return: инвертированное число
    """
    positive = False
    if number != 0:
        if number > 0:
            positive = True
        number = abs(number)
        while number % 10 == 0:
            number = number / 10
        string_number = (str(int(number)))[::-1]
        number = int(string_number)
        return number if positive else number*(-1)
    else:
        return 0
