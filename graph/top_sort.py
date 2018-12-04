# 拓扑排序

class KeyRoute:
    def __init__(self):
        pass

    def top_sort(self, Graph):
        """
        topologically sorted DAG
        :param Graph:
        :return:
        """
        in_degrees = dict((u, 0) for u in Graph)
        for u in Graph:
            for v in Graph[u]:
                in_degrees[v] += 1
        Q = [u for u in Graph if in_degrees[u] == 0]
        '''至少又一个入度为0,否则此图为有环图，不成立'''
        S = []
        while len(Q) > 0:
            u = Q.pop()
            S.append(u)
            for v in Graph[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    Q.append(v)
        return S


if __name__ == '__main__':
    G = {
        'a': 'f',
        'b': 'acdf',
        'c': 'd',
        'd': 'ef',
        'e': 'f',
        'f': ''
    }
    top = KeyRoute()
    S = top.top_sort(G)
    print(S)