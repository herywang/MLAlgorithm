from __future__ import print_function, absolute_import, division

class Sort:
    def __init__(self):
        pass
    
    def quick_sort(self, array, left, right):
        i = left + 1
        j = right
        tmp = array[left]
        while i<=j:
            while i<=j and array[i] <= tmp:
                i += 1
            while i<=j and array[j] >= tmp:
                j -= 1
            if i < j:
                self.__swap(array, i, j)
                i += 1
                j -= 1
        self.__swap(array, left, j)
        if left < j:
            self.quick_sort(array, left, j-1)
        if i < right:
            self.quick_sort(array, j+1, right)

    def __swap(self, arr, a, b):
        tmp = arr[a]
        arr[a] = arr[b]
        arr[b] = tmp
            
if __name__ == "__main__":
    arr = [3, 4, 5, 1, 2, 4, 5, 2, 23, 21, 0]
    sort = Sort()
    sort.quick_sort(array=arr, left=0, right=10)
    print(arr)

