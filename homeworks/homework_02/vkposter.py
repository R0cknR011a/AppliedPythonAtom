#!/usr/bin/env python
# coding: utf-8


#from homeworks.homework_02.heap import MaxHeap
#from homeworks.homework_02.fastmerger import FastSortedListMerger


class VKPoster:
    posts = {}
    followers = {}

    def __init__(self):
        pass

    def user_posted_post(self, user_id, post_id):
        '''
        Метод который вызывается когда пользователь user_id
        выложил пост post_id.
        :param user_id: id пользователя. Число.
        :param post_id: id поста. Число.
        :return: ничего
        '''
        self.posts[post_id] = user_id, [user_id]

    def user_read_post(self, user_id, post_id):
        '''
        Метод который вызывается когда пользователь user_id
        прочитал пост post_id.
        :param user_id: id пользователя. Число.
        :param post_id: id поста. Число.
        :return: ничего
        '''
        self.posts[post_id][1].append(user_id)

    def user_follow_for(self, follower_user_id, followee_user_id):
        '''
        Метод который вызывается когда пользователь follower_user_id
        подписался на пользователя followee_user_id.
        :param follower_user_id: id пользователя. Число.
        :param followee_user_id: id пользователя. Число.
        :return: ничего
        '''
        if self.followers[follower_user_id]:
            self.followers[follower_user_id].append(followee_user_id)
        else:
            self.followers[follower_user_id] = [followee_user_id]

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
        for i, j in self.posts.items():
            if k == 0:
                break
            if j[0] in self.followers[user_id]:
                m.append(i)
                k -= 1
        m.sort(reverse=True)
        return m



    def get_most_popular_posts(self, k):
        '''
        Метод который возвращает список k самых популярных постов за все время,
        остортированных по свежести.
        :param k: Сколько самых свежих популярных постов
        необходимо вывести. Число.
        :return: Список из post_id размером К из популярных постов. list
        '''
        m = []
        self.posts = sorted(self.posts.keys(), key=lambda x: x, reverse=True)
        for i in self.posts.keys():
            if k == 0:
                break
            m.append(i)
            k -= 1
        return m
