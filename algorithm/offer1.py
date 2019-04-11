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


# 二位数组查找.在一个二维数组中，每一行都是按照从左到右递增的顺去排列，每一列也是按照从上至下递增顺序排列。
# 请完成一个函数，输入这样的一个二维数组和整数，判断数组中是否含有该整数
def search_2_dim(array, n, *shape):
    if len(shape) != 2: raise Exception("shape must have 2 dimension!")
    if array is None or n is None: raise Exception("There is an error in array or n!")
    start_line = 0
    end_col = shape[1] - 1
    tmp = array[start_line][end_col]
    while n != tmp:
        if start_line == shape[0] or end_col < 0: return None
        if n > tmp:
            start_line += 1
        else:
            for i in range(end_col, -1, -1):
                if array[start_line][i] == n:
                    return (start_line, i)
            return None
        if n < tmp:
            end_col -= 1
        else:
            for i in range(shape[0]-1, -1, -1):
                if array[i][end_col] == n:
                    return (i, end_col)
            return None
        tmp = array[start_line][end_col]
    return (start_line, end_col)


if __name__ == '__main__':
    # head = ListNode(8, ListNode(7, ListNode(6, ListNode(5, ListNode(4, ListNode(3, ListNode(2, ListNode(1))))))))
    # findKthToTail(head, 3)
    # print(get_duplication([3,4,2,3,5], 5))
    print(search_2_dim([[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]], 12, 4, 4))

