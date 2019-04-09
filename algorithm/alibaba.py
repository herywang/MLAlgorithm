
"""
给定一个表达式，和一个数组，表达式中有n个缺失，在数组中选择n个分别填入缺失中，使得等式成立。
例子：
3_*2=_2
1 3 6 8
输出：
31*2=62
选出的数不能重复，并且n不确定。
"""

def search(exp, nums):
    if "_" not in exp or nums is None:
        exps = exp.split("=")
        if eval(exps[0]) == float(exps[1]):
            print(exps[0] + "=" + exps[1])
        return
    else:
        for num in nums:
            e = exp.replace("_", num, 1)
            ns = nums[:]
            ns.remove(num)
            search(e, ns)


if __name__ == '__main__':
    exp = input()
    nums = input().split(" ")
    search(exp, nums)