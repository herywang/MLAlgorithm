# 先来先服务调度算法实现
from __future__ import print_function, absolute_import, division

from operate_sys.priority_algorithm import PCB
from collections import deque
import time

class ShortTaskPriority:
    def __init__(self):
        self.progresses = deque()

    def __progresses_in(self):
        self.progresses.append(PCB(1, 10, 0))
        self.progresses.append(PCB(3, 3, 0.2))
        self.progresses.append(PCB(2, 2, 0.4))
        self.progresses.append(PCB(5, 5, 0.5))
        self.progresses.append(PCB(4, 1, 0.8))
        self.progresses.append(PCB(7, 9, 1.2))
        self.progresses.append(PCB(6, 6, 1.5))

    def progress_in(self):
        self.__progresses_in()


if __name__ == '__main__':
    short = ShortTaskPriority()
    short.progress_in()
    for progress in short.progresses:
        print(progress)
        progress.status=1
        print(progress, '\n')
        time.sleep(1)