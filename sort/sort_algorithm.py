from __future__ import print_function, absolute_import, division

class Sort:
    def __init__(self):
        self.heap = [-99999]
    
    def quick_sort(self, array, left, right):
        """
        :param array: An array that need to sort.
        :param left: left index.
        :param right: right index.
        :return: none return.
        """
        i = left + 1
        j = right
        tmp = array[left]
        while i<=j:
            while i<=j and array[i] < tmp:
                i += 1
            while i<=j and array[j] > tmp:
                j -= 1
            if i <= j:
                self.__swap(array, i, j)
                i += 1
                j -= 1
        self.__swap(array, left, j)
        if left < j:
            self.quick_sort(array, left, j-1)
        if i < right:
            self.quick_sort(array, j+1, right)

    def select_sort(self, array):
        """
        select sort
        :param array:
        :return: none
        """
        for i in range(len(array)):
            min = 9999
            min_index = 0
            for j in range(i, len(array)):
                if array[j] < min:
                    min = array[j]
                    min_index = j
            if min_index != i:
                self.__swap(array, i, min_index)

    def insert_sort(self, array):
        """
        Insert sort.
        :param array:
        :return:
        """
        for i in range(1, len(array)):
            j = i-1
            x = array[i]
            while x < array[j] and j>=0:
                array[j+1] = array[j]
                j -= 1
            array[j+1] = x

    def count_sort(self, array):
        min, max = self.__get_min_max(array)
        count = list([0] * (max - min + 1))


    def __swap(self, arr, a, b):
        tmp = arr[a]
        arr[a] = arr[b]
        arr[b] = tmp

    def __get_min_max(self, array):
        min = array[0]
        max = array[0]
        for i in array:
            if i < min:
                min = i
            if i > max:
                max = i
        return min, max

    def heap_append_value(self, value):
        length = len(self.heap) - 1
        self.heap.append(value)
        son = length+1
        father = son // 2
        while son != 1 and self.heap[father]>self.heap[son]:
            self.__swap(self.heap, son, father)
            son = father
            father = father // 2

    def heap_get_min(self):
        if len(self.heap)<=1:
            return None
        min = self.heap[1]
        self.heap[1] = self.heap.pop(len(self.heap)-1)
        self.heap_extract_sequence(1)
        return min

    def heap_extract_sequence(self, father):
        left = father * 2
        right = father * 2 + 1

        if left < len(self.heap) and right < len(self.heap):
            if self.heap[father] > self.heap[left] and self.heap[father] > self.heap[right]:
                if self.heap[left] < self.heap[right]:
                    self.__swap(self.heap, father, left)
                    father = left
                    self.heap_extract_sequence(father)
                else:
                    self.__swap(self.heap, father, right)
                    father = right
                    self.heap_extract_sequence(father)
            elif self.heap[father] > self.heap[left]:
                self.__swap(self.heap, father, left)
                father = left
                self.heap_extract_sequence(father)
            elif self.heap[father] > self.heap[right]:
                self.__swap(self.heap, father, right)
                father = right
                self.heap_extract_sequence(father)
        elif left < len(self.heap):
            if self.heap[father] > self.heap[left]:
                self.__swap(self.heap, father, left)
                father = left
                self.heap_extract_sequence(father)
        elif right < len(self.heap):
            self.__swap(self.heap, father, right)
            father = right
            self.heap_extract_sequence(father)

    def update_heap(self, index, value):
        try:
            self.heap[index] = value
            self.__update(index)
        except:
            raise IndexError("Array index out of bound!")

    def __update(self, index):
        father = index // 2
        left = father * 2
        right = father * 2 + 1
        if self.heap[index] < self.heap[father]:
            self.__swap(self.heap, index, father)
            self.__update(father)
        else:
            self.heap_extract_sequence(index)

if __name__ == "__main__":
    arr = [3, 4, 5, 1, 2, 4, 5]
    sort = Sort()
    # sort.quick_sort(array=arr, left=0, right=len(arr)-1)
    # sort.insert_sort(arr)
    # print(arr)
    for i in range(len(arr)):
        sort.heap_append_value(arr[i])
        print(sort.heap)
    # for i in range(5):
    #     min = sort.heap_get_min()
    #     print(min, sort.heap)
    print("update heap=========")
    sort.update_heap(1, 10)
    print(sort.heap)

