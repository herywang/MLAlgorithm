# 拓扑排序
import copy

class KeyRoute:
    def __init__(self):
        pass

    @staticmethod
    def top_sort(Graph):
        """
        topologically sorted DAG
        :param Graph:
        :return:
        """
        in_degrees = dict((u, 0) for u in Graph.keys())
        for u in Graph.keys():
            for v in Graph[u].keys():
                if v in in_degrees:
                    in_degrees[v] += 1
                else:
                    in_degrees[v] = 0
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

    @staticmethod
    def event_earliest_time(vnum, Graph, topseq):
        '''
        @param vnum: 节点数量
        @param graph: 图
        @param topseq: 拓扑排序序列
        '''
        ee = dict((e, 0) for e in topseq)
        for i in topseq:
            for j in Graph[i]:
                if ee[i] + Graph[i][j] > ee[j]:
                    ee[j] = ee[i] + Graph[i][j]
        return ee

    @staticmethod
    def event_latest_time(Graph, topseq, eelast):
        tmp_topseq = copy.deepcopy(topseq)
        el = dict((e, eelast) for e in tmp_topseq)
        for i in range(len(topseq)-1, -1, -1):
            k = topseq[i]
            for key, value in Graph.items():
                if k in value.keys():
                    if el[k] - Graph[key][k] < el[key]:
                        el[key] = el[k] - Graph[key][k]
        return el

    @staticmethod
    def critical_path(Graph, ee, el, topseq):
        sequence = []
        for element in topseq:
            if ee[element] == el[element]:
                sequence.append(element)
        return sequence

if __name__ == '__main__':
    G = {
        'A': {'B': 6, 'C': 4, 'D': 5},
        'B': {'E': 1},
        'C': {'E': 1},
        'D': {'F': 2},
        'E': {'G': 5, 'H': 7},
        'F': {'G': 4},
        'G': {'I': 4},
        'H': {'I': 2},
        'I': {}
    }
    S = KeyRoute.top_sort(G)
    print(S)
    ete = KeyRoute.event_earliest_time(9, G, S)
    print(ete)
    elt = KeyRoute.event_latest_time(G, S, ete[S[-1]])
    print(elt)
    sequence = KeyRoute.critical_path(G, ete, elt, S)
    print(sequence)