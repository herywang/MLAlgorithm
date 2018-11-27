from __future__ import print_function, absolute_import, division

class Sort:
    def __init__(self):
        pass
    
    def quick_sort(self, array, left, right):
        i = left
        j = right
        tmp = array[(left + right) // 2]
        while i<=j:
            while i<j and array[i] <= tmp:
                i += 1
            while i<j and array[j] >= tmp:
                j -= 1
            if i <= j:
                t = array[i]
                array[i] = array[j]
                array[j] = t
                i += 1
                j -= 1
        if left < j:
            self.quick_sort(array, left, j)
        if i < right:
            self.quick_sort(array, i, right)
            
if __name__ == "__main__":
    arr = [3, 4, 5, 1, 2, 4, 5, 2, 23, 21, 0]
    sort = Sort()
    sort.quick_sort(array=arr, left=0, right=10)
    print(arr)

