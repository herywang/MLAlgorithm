# dijkstra算法求解最短路径
import heapq
import math

class Dijkstra(object):

    def dijkstra(self, src):
        graph = {
            'A': {'B': 5, 'C': 1},
            'B': {'A': 5, 'C': 2, 'D': 1},
            'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
            'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
            'E': {'C': 8, 'D': 3},
            'F': {'D': 6}
        }
        pqueue = []
        heapq.heappush(pqueue, (0, src))
        seen = set()
        parent = {src: None}
        distance = self.init_distance(graph, src)

        while len(pqueue) > 0:
            pair = heapq.heappop(pqueue)
            dist = pair[0]
            vertex = pair[1]
            seen.add(vertex)
            nodes = graph[vertex].keys() #获取邻接点
            for w in nodes:
                if w not in seen:
                    if dist + graph[vertex][w] < distance[w]:
                        heapq.heappush(pqueue, (dist + graph[vertex][w], w))
                        parent[w] = vertex
                        distance[w] = dist + graph[vertex][w]
        return parent, distance

    @staticmethod
    def init_distance(graph, src):
        distance = {src: 0}
        for vertex in graph:
            if vertex != src:
                distance[vertex] = math.inf
        return distance

if __name__ == "__main__":
    path = Dijkstra()
    parent, distance = path.dijkstra('B')
    print(parent, '\n\n', distance)
