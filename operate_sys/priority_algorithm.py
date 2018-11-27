# priority algorithm.优先级调度算法
from collections import deque
import time
import random

class PCB:

    def __init__(self, priority, hold_time, come_time,  status = 0):
        """
        :param status: PCB 进程控制块状态：0为等待态， 1为执行态
        :param priority: 优先级别：0， 1， 2， 3， 4， ...
        :param hold_time: 处理机占用时间
        :param come_time: 进入进程队列时间
        """
        self.status = status
        self.priority = priority
        self.hold_time = hold_time
        self.come_time = come_time
        self.progress_name = random.randint(1000, 9999)

    def __str__(self):
        return str(self.progress_name) + "\t" + str(self.priority) + "\t\t\t" + \
               str(self.come_time) + "\t\t\t" + str(self.hold_time) + "\t\t\t\t" + \
               str('就绪态' if self.status == 0 else '执行态')

class PriorityAlgorithm:

    def __init__(self):
        self.progresses = deque()
        self.__progresses_in()
        self.max_index = -1

    def __progresses_in(self):
        self.progresses.append(PCB(1, 10, 0))
        self.progresses.append(PCB(3, 3, 0.2))
        self.progresses.append(PCB(2, 2, 0.4))
        self.progresses.append(PCB(5, 5, 0.5))
        self.progresses.append(PCB(4, 1, 0.8))
        self.progresses.append(PCB(7, 9, 1.2))
        self.progresses.append(PCB(6, 6, 1.5))

    def print_all_progresses(self):
        print("当前队列中就绪的进程有：")
        print(str('name') + "\t" + str('priority') + "\t" + \
               str('come_time') + "\t" + str('execute_time') + "\t" + \
               str('status'))
        for i in range(len(self.progresses)):
            print(self.progresses[i])

    def execute_progress(self):
        while len(self.progresses) > 0:
            for i in range(len(self.progresses)):
                self.max_index = self.get_max_priority_index()
                self.progresses[self.max_index].hold_time -= 1
                if self.progresses[self.max_index].hold_time <= 0:
                    self.progresses.remove(self.progresses[self.max_index])
                self.print_all_progresses()
                time.sleep(1)
            self.__change_priority()
            print("===============================一次时间片轮回后结果：================================")
            self.print_all_progresses()
            time.sleep(1)

    def get_max_priority_index(self):
        max_index = 0
        for i in range(len(self.progresses)):
            if self.progresses[i].priority > self.progresses[max_index].priority:
                max_index = i
        self.progresses[max_index].priority = self.progresses[max_index].priority - 10
        return max_index

    def __change_priority(self):
        for i in range(len(self.progresses)):
            self.progresses[i].priority += 10

if __name__ == "__main__":
    priority = PriorityAlgorithm()
    print("初始化队列进程：===================================")
    priority.print_all_progresses()
    print("===================end============================")
    priority.execute_progress()





