class ListNode(object):
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


def findKthToTail(head, k):
    if head is None:
        raise Exception("head is None!")
    if k == 0:
        raise Exception("The value of K can not be 0!")
    ahead = head
    for i in range(k - 2):
        if ahead is not None:
            ahead = ahead.next
        else:
            raise Exception("There is an Error!")
    behind = head
    while ahead is not None:
        ahead = ahead.next
        behind = behind.next
    print(behind.data)


# 找出重复数字
def get_duplication(a, length):
    if a is None or length <= 0: raise Exception("a is None or length Error!")
    for i in range(length):
        while a[i] != i:
            if a[i] == a[a[i]]:
                return a[i]
            else:
                t = a[i]
                a[i] = a[a[i]]
                a[a[i]] = t


if __name__ == '__main__':
    # head = ListNode(8, ListNode(7, ListNode(6, ListNode(5, ListNode(4, ListNode(3, ListNode(2, ListNode(1))))))))
    # findKthToTail(head, 3)
    print(get_duplication([3,4,2,3,5], 5))