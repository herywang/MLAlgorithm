# 暴力求解法：1.简单枚举

def division():
    '''
    输入整数n，按从小到大的顺序输出所有形如 abcde/fjhij = n的表达式， 其中a~j恰好为数字0~9的一个排列
    :return:
    '''
    n = input("请输入一个正整数n, 2<=n<=79\n")
    for a in range(1, 10):
        for b in range(10):
            if a == b:
                continue
            else:
                for c in range(10):
                    if c==a or c==b:
                        continue
                    else:
                        for d in range(10):
                            if d==c or d==b or d==a:
                                continue
                            else:
                                for e in range(10):
                                    if e==d or e==c or e==b or e==a:
                                        continue
                                    else:
                                        value = a*10000 + b*1000 + c*100 + d*10 + e
                                        if value % int(n) == 0:
                                            div = value // int(n)
                                            str_value = str(value)
                                            str_div = str(div)

                                            set_str_div = set(str_div)
                                            set_str_value = set(str_value)
                                            if '0' in set_str_value or '0' in set_str_div:
                                                sets = set_str_div | set_str_value
                                            else:
                                                set_str_div = set('0') | set_str_div
                                                sets = set_str_div | set_str_value

                                            if len(sets) == 10:
                                                print(str(value) + "\t/\t0" + str_div + "\t=\t" + str(n))
                                            else:
                                                continue

def maximum_product():
    '''
    最大乘积。输入n个元素组成的序列S,你需要找出一个乘积最大的连续子序列， 如果这个最大的成绩不是正数， 应输出0,表示无解。
    :return:
    '''
    n = input("请输入序列长度：\n")
    si = []
    for i in range(int(n)):
        sii = input("元素值：")
        si.append(int(sii))
    max = 0
    tmp_max = 1
    for i in range(int(n)-1, -1, -1):
        for j in range(i+1):
            tmp_max *= si[j]
        if tmp_max > max:
            max = tmp_max
        tmp_max = 1
    for i in range(int(n)):
        for j in range(i, int(n)):
            tmp_max *= si[j]
        if tmp_max > max:
            max = tmp_max
        tmp_max = 1

    print(max)

def fraction_again():
    '''
    分数拆分，输入正整数k, 找到所有x>=y, 使得 1/k = 1/x + 1/y
    :return:
    '''
    k = input("请输入k值：")
    for y in range(1, 2*int(k)):
        x = y
        while True:
            if x*y // (x+y) == k and x*y % (x+y) == 0:
                print("1/k = 1/" + str(x) + " 1/" + str(y))
                break
            x += 1
if __name__ == '__main__':
    # division()
    # maximum_product()
    fraction_again()
