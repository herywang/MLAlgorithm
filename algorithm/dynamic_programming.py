import numpy as np

sub_fib = np.ones((100), np.int) * -1


# 自顶向下的备忘录法
def fib(n, memory):
    if memory[n] != -1:
        return memory[n]
    if n <= 2:
        memory[n] = 1
    else:
        memory[n] = fib(n - 1, memory) + fib(n - 2, memory)
    return memory[n]


# 自底向上动态规划
# 备忘录法还是利用了递归，上面算法不管怎样，计算fib（6）的时候
# 最后还是要计算出fib（1），fib（2），fib（3）……,那么何不先计算出
# fib（1），fib（2），fib（3）……,呢？这也就是动态规划的核心，
# 先计算子问题，再由子问题计算父问题。

def fib_1(n):
    if n <= 0:
        return n
    Memo = np.ones((100), np.int) * -1
    Memo[0] = 0
    Memo[1] = 1
    for i in range(2, n + 1):
        Memo[i] = Memo[i - 1] + Memo[i - 2]
    return Memo[n]


# 自底向上方法也是利用数组保存了先计算的值，为
# 后面的调用服务。观察参与循环的只有 i，i-1 , i-2
# 三项，因此该方法的空间可以进一步的压缩如下。
def fib_2(n):
    if n <= 0:
        return n
    memo_i_2 = 0
    memo_i_1 = 1
    memo_i = 1
    for i in range(2, n + 1, 1):
        memo_i = memo_i_1 + memo_i_2
        memo_i_2 = memo_i_1
        memo_i_1 = memo_i
    return memo_i


# 动态规划初步：数字三角形问题：有一个由非负整数组成的数字三角形，
# 第一行只有一个参数，除了最下行之外，每一个数的左下方和右下方各
# 有一个数，从第一行的数开始，走到最下面，如何走才能使得沿途经过
# 的数字之和最大。
# 状态转移方程：d(i,j) = a(i,j) + max(d(i+1, j), d(i+1, j+1))
# 1. 递归算法
def solve(a, i, j, n):
    return a[i, j] + (0 if i == n else max(solve(a, i + 1, j, n), solve(a, i + 1, j + 1, n)))

# 2. 递推算法
# def solve2()

if __name__ == '__main__':
    # print(fib(8, sub_fib))
    # print(fib_1(9))
    # print(fib_2(10))

    a = np.zeros((4, 4))
    a[0, 0] = 1
    a[1, 0], a[1, 1] = 3, 2
    a[2, 0], a[2, 1], a[2, 2] = 4, 10, 1
    a[3, 0], a[3, 1], a[3, 2], a[3, 3] = 4, 3, 2, 20
    print(solve(a, 0, 0, 3))
