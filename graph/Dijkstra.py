# dijkstra算法求解最短路径
import numpy as np

class Dijkstra:
    def __init__(self):
        self.MAX = 999
        self.adjacent_mat = np.matrix('0 2 1 3 0 0 0;'
                                      '0 0 0 1 0 0 0;'
                                      '0 0 0 0 4 0 0;'
                                      '0 0 0 0 1 3 0;'
                                      '0 0 0 0 0 0 2;'
                                      '0 0 0 0 0 0 2;'
                                      '0 0 0 0 0 0 0')
        self.nodes = [i for i in range(len(self.adjacent_mat))]
        self.visited = []


    def dijkstra(self, src=0):
        print('邻接矩阵:\n', self.adjacent_mat)
        if src in self.nodes:
            self.nodes.remove(src)
            self.visited.append(src)
        else:
            return None
        distance = {}
        for i in self.nodes:
            distance[i] = self.adjacent_mat[src, i] if self.adjacent_mat[src, i] != 0 else self.MAX
        print(src, '\n', distance)
        path = {src:{src:[]}}
        global k, pre
        k = pre = src
        while self.nodes:
            mid_distances = float('inf')
            for v in self.visited:
                for d in self.nodes:
                    new_instance = self.adjacent_mat[src, v] + self.adjacent_mat[v, d]
                    if new_instance < mid_distances:
                        mid_distances = new_instance
                        self.adjacent_mat[src, d] = new_instance
                        k=d
                        pre=v
            distance[k] = mid_distances
            path[src][k] = [i for i in path[src][pre]]
            path[src][k].append(k)
            self.visited.append(k)
            self.nodes.remove(k)
            print(self.visited, self.nodes)
        return distance, path

if __name__ == "__main__":
    path = Dijkstra()
    distance, path = path.dijkstra()
    print(distance, path)
