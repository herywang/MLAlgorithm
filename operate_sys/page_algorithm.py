# 页面置换算法
from collections import deque

class Node(object):
    def __init__(self, page):
        self.page = page
        self.count = 1 # 被访问次数
    
    def __str__(self):
        return str(self.page)

class PageAlgorithm(object):
    def __init__(self, phy_size=3):
        self.length = 0
        self.phy_size = phy_size
        self.queue = deque() # FIFO算法数据结构
        self.lru_structure = []
    
    def FIFO(self):
        while True:
            page_num = input("请输入要访问的页面号：")
            if page_num == -1 or page_num == '-1':
                break
            else:
                if (not self._in_queue(page_num)) and (self.length < self.phy_size):
                    self.queue.append(Node(page_num))
                    self.length += 1
                    print(str(page_num) + " 页 缺页， 页面加入物理快，当前物理快中的页面有：",end='')
                    self._print_queue()
                elif(not self._in_queue(page_num)) and (self.length >= self.phy_size):
                    self.queue.popleft()
                    self.queue.append(Node(page_num))
                    print(str(page_num) + "页 缺页， 执行FIFO算法， 当前物理快中的页面有：", end='')
                    self._print_queue()
                else:
                    print(str(page_num) + "页 不缺页， 当前物理快中的页面有：", end='')
                    self._print_queue()
            print()
    def LRU(self):
        while True:
            page_num = input("请输入要访问的页面号：")
            if page_num == -1 or page_num == '-1':
                break
            else:
                if (not self._in_lru_structure(page_num)) and (self.length < self.phy_size):
                    self.lru_structure.append(Node(page_num))
                    self.length += 1
                    print(str(page_num) + " 页 缺页， 页面加入物理快，当前物理快中的页面有：\n页面号\t访问次数")
                    self._print_lry_structure()

                elif(not self._in_lru_structure(page_num)) and (self.length >= self.phy_size):
                    min_indicies = self._min_count_indices()
                    self.lru_structure.pop(min_indicies)
                    self.lru_structure.append(Node(page_num))
                    print(str(page_num) + "页 缺页， 执行LRU算法， 当前物理快中的页面有：\n页面号\t访问次数")
                    self._print_lry_structure()

                else:
                    self._count_add_one(page_num)
                    print(str(page_num) + "页 不缺页， 当前物理快中的页面有：\n页面号\t访问次数")
                    self._print_lry_structure()
            print()

    def _in_queue(self, page_num):
        for node in self.queue:
            if page_num == node.page:
                return True
        return False
    
    def _in_lru_structure(self, page_num):
        for node in self.lru_structure:
            if page_num == node.page:
                return True
        return False
        
    def _min_count_indices(self):
        min = float('inf')
        min_indicies = -1
        for i in range(len(self.lru_structure)):
            if self.lru_structure[i].count < min:
                min = self.lru_structure[i].count
                min_indicies = i
        return min_indicies

    def _count_add_one(self, page_num):
        for node in self.lru_structure:
            if page_num == node.page:
                node.count += 1
                break

    def _print_queue(self):
        for i in self.queue:
            print(str(i.page) + "\t" + str(i.count))

    def _print_lry_structure(self):
        for i in self.lru_structure:
            print(str(i.page) + "\t" + str(i.count))
if __name__ == '__main__':
    page = PageAlgorithm(3)
    # page.FIFO()
    page.LRU()
