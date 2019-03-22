#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Process, Manager
import os


def word_count_inference(path_to_dir):
    '''
    Метод, считающий количество слов в каждом файле из директории
    и суммарное количество слов.
    Слово - все, что угодно через пробел, пустая строка "" словом не считается,
    пробельный символ " " словом не считается. Все остальное считается.
    Решение должно быть многопроцессным. Общение через очереди.
    :param path_to_dir: путь до директории с файлами
    :return: словарь, где ключ - имя файла, значение - число слов +
        специальный ключ "total" для суммы слов во всех файлах
    '''
    files = [i for i in os.listdir(path_to_dir)]
    print(files)
    manager = Manager()
    dct = manager.dict()
    processes = []
    for i in files:
        proc = Process(target=counter, args=(path_to_dir + '/' + i, dct))
        processes.append(proc)
        proc.start()
    for i in processes:
        i.join()
    dct['total'] = sum(dct.values())
    print(dct)
    return dct


def counter(filename, dct):
    data = 0
    with open(filename, encoding='utf8') as f:
        for i in f:
            data += len(i.split())
    dct[filename.split('/')[-1]] = data


