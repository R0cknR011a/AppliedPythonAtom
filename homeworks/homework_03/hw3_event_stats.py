#!/usr/bin/env python
# coding: utf-8


class TEventStats:
    FIVE_MIN = 300

    def __init__(self):
        # TODO: реализовать метод
        self.dict = {}

    def register_event(self, user_id, time):
        """
        Этот метод регистрирует событие активности пользователя.
        :param user_id: идентификатор пользователя
        :param time: время (timestamp)
        :return: None
        """
        # TODO: реализовать метод
        if user_id in self.dict.keys():
            self.dict[user_id].append(time)
        else:
            self.dict[user_id] = [time]

    def query(self, count, time):
        """
        Этот метод отвечает на запросы.
        Возвращает количество пользователей, которые за последние 5 минут
        (на полуинтервале времени (time - 5 min, time]), совершили ровно count действий
        :param count: количество действий
        :param time: время для рассчета интервала
        :return: activity_count: int
        """
        # TODO: реализовать метод
        result = []
        for i, j in self.dict.items():
            cnt = sum(time - 300 < x < time for x in j)
            if cnt == 0 and count == 0:
                continue
            if cnt == count:
                result.append(i)
        return len(result)
