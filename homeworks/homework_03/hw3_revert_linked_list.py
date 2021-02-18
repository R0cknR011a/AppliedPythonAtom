#!/usr/bin/env python
# coding: utf-8


def revert_linked_list(head):
    """
    A -> B -> C should become: C -> B -> A
    :param head: LLNode
    :return: new_head: LLNode
    """
    # TODO: реализовать функцию
    previous = None
    current = head
    while current:
        next = current.next_node
        current.next_node = previous
        previous = current
        current = next
    return previous
