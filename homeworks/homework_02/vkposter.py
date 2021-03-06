#!/usr/bin/env python
# coding: utf-8


from homeworks.homework_02.heap import MaxHeap
from homeworks.homework_02.fastmerger import FastSortedListMerger


class VKPoster:

    def __init__(self):
        self.posts = {}
        self.followers = {}

    def user_posted_post(self, user_id, post_id):
        '''
        Метод который вызывается когда пользователь user_id
        выложил пост post_id.
        :param user_id: id пользователя. Число.
        :param post_id: id поста. Число.
        :return: ничего
        '''
        self.posts[post_id] = user_id, []

    def user_read_post(self, user_id, post_id):
        '''
        Метод который вызывается когда пользователь user_id
        прочитал пост post_id.
        :param user_id: id пользователя. Число.
        :param post_id: id поста. Число.
        :return: ничего
        '''
        if post_id in self.posts:
            if user_id not in self.posts[post_id][1]:
                self.posts[post_id][1].append(user_id)
        else:
            self.posts[post_id] = '', [user_id]

    def user_follow_for(self, follower_user_id, followee_user_id):
        '''
        Метод который вызывается когда пользователь follower_user_id
        подписался на пользователя followee_user_id.
        :param follower_user_id: id пользователя. Число.
        :param followee_user_id: id пользователя. Число.
        :return: ничего
        '''
        if follower_user_id not in self.followers:
            self.followers[follower_user_id] = [followee_user_id]
        else:
            self.followers[follower_user_id].append(followee_user_id)

    def get_recent_posts(self, user_id, k):
        '''
        Метод который вызывается когда пользователь user_id
        запрашивает k свежих постов людей на которых он подписан.
        :param user_id: id пользователя. Число.
        :param k: Сколько самых свежих постов необходимо вывести. Число.
        :return: Список из post_id размером К из свежих постов в
        ленте пользователя. list
        '''
        m = []
        for author in self.followers[user_id]:
            m.append([i for i in self.posts.keys()
                      if self.posts[i][0] == author])
        return FastSortedListMerger.merge_first_k(m, k)

    def get_most_popular_posts(self, k):
        '''
        Метод который возвращает список k самых популярных постов за все время,
        остортированных по свежести.
        :param k: Сколько самых свежих популярных постов
        необходимо вывести. Число.
        :return: Список из post_id размером К из популярных постов. list
        '''
        m = MaxHeap([])
        result = []
        while len(m.heap) < len(self.posts):
            for i, j in self.posts.items():
                m.add((len(j[1]), i))
        for i in range(k):
            result.append(m.extract_maximum()[1])
        return result
